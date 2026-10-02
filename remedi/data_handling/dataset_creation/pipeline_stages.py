import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
from ase import Atoms
from tqdm import tqdm

from remedi.configuration.dataset_config import (
    DatasetCreationConfig,
    PhysicochemicalDescriptorStageConfig,
)
from remedi.data_handling import physchem
from remedi.data_handling.chemistry.conformers import (
    ConformerTimingRecord,
    EmbedResult,
    embed_many,
    write_timings_jsonl,
)
from remedi.data_handling.chemistry.geometry import GeometryLimits, geometry_violations
from remedi.data_handling.chemistry.smiles_filter import (
    SmilesFilterConfig,
    filter_smiles,
)
from remedi.data_handling.dataset_creation.build_stats import LoadStats, StageStats
from remedi.data_handling.dataset_creation.loading_batch import (
    DataBatch,
    InputBatch,
    RegressionData,
    SmilesData,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID


def _slice_regression(rd: RegressionData, idx: np.ndarray) -> RegressionData:
    """Reindex a RegressionData by row indices (system axis).

    ``idx`` may be empty; numpy handles the zero-length slice naturally.
    Atom-axis targets stay unsupported (matches the existing pipeline).
    """
    new: dict = {}
    if rd.targets_system is not None:
        assert rd.mask_system is not None, "targets_system without mask_system"
        new["targets_system"] = rd.targets_system[idx, :]
        new["mask_system"] = rd.mask_system[idx, :]
    if rd.targets_atom is not None:
        raise NotImplementedError(
            "Atom-axis target reindexing inside filter stages is not implemented"
        )
    if rd.split is not None:
        new["split"] = rd.split[idx]
    return RegressionData(**new)


def _reindex_in_place(input_batch: InputBatch, kept_idx: list[int]) -> None:
    """Shrink every per-system list/array on ``input_batch`` to ``kept_idx``.

    Used by the two filter stages so they share the parallel-array bookkeeping.
    Touches ``structure_ids``, ``smiles``, ``raw_smiles``, ``molecules``,
    ``total_charge``, ``multiplicity``, and ``regression_data``.
    """
    input_batch.structure_ids = [input_batch.structure_ids[i] for i in kept_idx]
    if input_batch.molecules is not None:
        input_batch.molecules = [input_batch.molecules[i] for i in kept_idx]
    if input_batch.smiles is not None:
        input_batch.smiles = [input_batch.smiles[i] for i in kept_idx]
    if input_batch.raw_smiles is not None:
        input_batch.raw_smiles = [input_batch.raw_smiles[i] for i in kept_idx]
    if input_batch.total_charge is not None:
        input_batch.total_charge = [input_batch.total_charge[i] for i in kept_idx]
    if input_batch.multiplicity is not None:
        input_batch.multiplicity = [input_batch.multiplicity[i] for i in kept_idx]
    if input_batch.regression_data is not None:
        idx_arr = np.asarray(kept_idx, dtype=np.int64)
        input_batch.regression_data = _slice_regression(
            input_batch.regression_data, idx_arr
        )


class PipelineStage(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self, input_batch: InputBatch, data_batch: DataBatch | None = None
    ) -> tuple[InputBatch, DataBatch | None]:
        pass


type Pipeline = list[PipelineStage]


def _per_system_charge_multiplicity(
    input_batch: InputBatch, n_systems: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_batch.total_charge is None:
        charge = torch.zeros(n_systems, dtype=dtype)
    else:
        charge = torch.as_tensor(input_batch.total_charge, dtype=dtype)
    if input_batch.multiplicity is None:
        mult = torch.zeros(n_systems, dtype=dtype)
    else:
        mult = torch.as_tensor(input_batch.multiplicity, dtype=dtype)

    if charge.shape[0] != n_systems or mult.shape[0] != n_systems:
        raise ValueError(
            f"total_charge / multiplicity must have length {n_systems}, "
            f"got {charge.shape[0]} / {mult.shape[0]}"
        )
    return charge, mult


def _stack_atoms(
    molecules: list[Atoms], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_counts = [len(m) for m in molecules]
    positions = torch.from_numpy(
        np.concatenate([m.get_positions() for m in molecules], axis=0)
    ).to(dtype=dtype)
    atomic_numbers = torch.from_numpy(
        np.concatenate([m.get_atomic_numbers() for m in molecules], axis=0)
    ).to(dtype=torch.long)
    system_idx = torch.repeat_interleave(
        torch.arange(len(molecules), dtype=torch.long),
        torch.tensor(atom_counts, dtype=torch.long),
    )
    return positions, atomic_numbers, system_idx


def _physchem_worker(
    names: list[str], item: tuple[str | None, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Top-level worker (picklable): one structure -> (values, finite-mask)."""
    iso_smiles, atomic_numbers, positions = item
    return physchem.compute_row(names, iso_smiles, atomic_numbers, positions)


class PhysicochemicalDescriptorStage(PipelineStage):
    """Per-structure cheap RDKit descriptors written as ``targets_system`` columns.

    Provenance-independent: place it after whatever populates
    ``input_batch.molecules`` (after ``FilterAtomsStage`` for loaded datasets,
    after ``ConformerGenerationStage`` for generated ones) and before
    ``CopyDataStage``. 2D descriptors are computed from ``input_batch.smiles``;
    3D descriptors (SASA) from the loaded geometry. Missing SMILES masks the 2D
    columns; geometry-perception failure masks the 3D columns. Column order
    follows ``config.descriptor_names`` and must match ``config.to_task_set()``.
    """

    def __init__(
        self,
        config: PhysicochemicalDescriptorStageConfig,
        num_workers: int | None = None,
    ):
        self.config = config
        self.names = list(config.descriptor_names)
        # Validate names up front (raises KeyError on unknown descriptor).
        physchem.split_names(self.names)
        self.num_workers = os.cpu_count() if num_workers is None else int(num_workers)

    def __call__(self, input_batch: InputBatch, data_batch):
        molecules = input_batch.molecules
        if molecules is None:
            raise ValueError(
                "PhysicochemicalDescriptorStage requires populated "
                "input_batch.molecules; place it after the stage that produces "
                "them (FilterAtomsStage or ConformerGenerationStage)."
            )

        n = len(molecules)
        smiles = input_batch.smiles
        items = [
            (
                smiles[i].isomeric_smiles if smiles is not None else None,
                molecules[i].get_atomic_numbers(),
                molecules[i].get_positions(),
            )
            for i in range(n)
        ]

        worker = partial(_physchem_worker, self.names)
        if self.num_workers and self.num_workers > 1 and n > 1:
            chunksize = max(1, n // (self.num_workers * 4))
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                rows = list(ex.map(worker, items, chunksize=chunksize))
        else:
            rows = [worker(it) for it in items]

        n_cols = len(self.names)
        targets = np.zeros((n, n_cols), dtype=np.float32)
        masks = np.zeros((n, n_cols), dtype=np.uint8)
        for i, (vals, mask) in enumerate(rows):
            masks[i] = mask
            # Keep only valid entries; zero masked ones so a forgotten mask never
            # propagates NaN downstream (the mask stays authoritative).
            targets[i] = np.where(mask.astype(bool), vals, 0.0).astype(np.float32)

        input_batch.regression_data = self._merge_targets(
            input_batch.regression_data, targets, masks
        )
        return input_batch, data_batch

    @staticmethod
    def _merge_targets(
        rd: RegressionData | None, targets: np.ndarray, masks: np.ndarray
    ) -> RegressionData:
        """Set the physchem columns, appending if prior system targets exist."""
        if rd is None:
            return RegressionData(targets_system=targets, mask_system=masks)
        if rd.targets_system is None:
            return RegressionData(
                targets_system=targets,
                mask_system=masks,
                targets_atom=rd.targets_atom,
                mask_atom=rd.mask_atom,
                split=rd.split,
            )
        return RegressionData(
            targets_system=np.concatenate([rd.targets_system, targets], axis=1),
            mask_system=np.concatenate([rd.mask_system, masks], axis=1),
            targets_atom=rd.targets_atom,
            mask_atom=rd.mask_atom,
            split=rd.split,
        )


class CopyDataStage(PipelineStage):
    def __init__(self, dtype):
        self._dtype = dtype

    def __call__(self, input_batch: InputBatch, output_batch):
        assert output_batch is None

        positions, atomic_numbers, system_idx = _stack_atoms(
            input_batch.molecules, self._dtype
        )

        n_systems = len(input_batch.molecules)
        charge, mult = _per_system_charge_multiplicity(
            input_batch, n_systems, self._dtype
        )

        output_batch = DataBatch(
            atomic_positions=positions,
            atomic_numbers=atomic_numbers,
            systems_index=system_idx,
            smiles_data=input_batch.smiles,
            structure_ids=input_batch.structure_ids,
            total_charge=charge,
            multiplicity=mult,
            regression_data=input_batch.regression_data,
        )

        return input_batch, output_batch


class ConformerGenerationStage(PipelineStage):
    """ETKDG + MMFF embedding of every SMILES in the batch, via ``embed_many``.

    Each molecule yields up to ``config.conformers.n_conformers`` structures;
    per-system arrays (charge, multiplicity, regression data) are replicated
    onto them. Molecules are emitted in input order.
    """

    def __init__(
        self,
        dataset_creation_config: DatasetCreationConfig,
        n_workers: int | None = None,
    ):
        self.config = dataset_creation_config
        self.n_workers = n_workers
        # Per-stage running totals across all batches; orchestrator reads this
        # at finalize for the build summary.
        self.stats = StageStats()
        # Per-molecule timing records, dumped to JSONL by ``flush_timings``.
        self._timing_records: list[ConformerTimingRecord] = []

    def __call__(
        self, input_batch: InputBatch, data_batch: DataBatch | None = None
    ) -> tuple[InputBatch, DataBatch | None]:
        if input_batch.smiles is None:
            raise ValueError(
                "ConformerGenerationStage requires input_batch.smiles; place it "
                "after FilterMoleculeStage"
            )
        smiles = input_batch.smiles
        self.stats.n_attempted += len(smiles)
        results = dict(
            embed_many(
                {
                    index: smiles_data.isomeric_smiles
                    for index, smiles_data in enumerate(smiles)
                },
                self.config.conformers,
                n_workers=self.n_workers,
            )
        )

        molecules: list[Atoms] = []
        smiles_per_structure: list[SmilesData] = []
        structure_ids: list[StructureID] = []
        parent_indices: list[int] = []
        for index, smiles_data in enumerate(smiles):
            result = results[index]
            self._timing_records.append(result.timing)
            if not result.succeeded:
                self._count_failure(smiles_data.isomeric_smiles, result)
                continue
            assert result.positions is not None and result.atomic_numbers is not None
            parent = input_batch.structure_ids[index]
            for positions in result.positions:
                molecules.append(
                    Atoms(
                        positions=positions,
                        numbers=result.atomic_numbers,
                        pbc=[0, 0, 0],
                    )
                )
                smiles_per_structure.append(smiles_data)
                structure_ids.append(
                    StructureID(
                        structure_id=-1,
                        molecule_id=parent.molecule_id,
                        stereoisomer_id=parent.stereoisomer_id,
                    )
                )
                parent_indices.append(index)
            self.stats.n_emitted += 1

        input_batch.molecules = molecules
        input_batch.smiles = smiles_per_structure
        input_batch.structure_ids = structure_ids
        if input_batch.total_charge is not None:
            input_batch.total_charge = [
                input_batch.total_charge[i] for i in parent_indices
            ]
        if input_batch.multiplicity is not None:
            input_batch.multiplicity = [
                input_batch.multiplicity[i] for i in parent_indices
            ]
        if input_batch.regression_data is not None:
            input_batch.regression_data = _slice_regression(
                input_batch.regression_data, np.asarray(parent_indices, dtype=np.int64)
            )
        return input_batch, data_batch

    def _count_failure(self, isomeric_smiles: str, result: EmbedResult) -> None:
        # Parse failures are "value_error"; embedding, relaxation and worker
        # failures count as other errors.
        if result.timing.status == "value_error":
            self.stats.n_value_errors += 1
            tqdm.write(f"[skip] {isomeric_smiles}: {result.timing.error_msg}")
        else:
            self.stats.n_other_errors += 1
            tqdm.write(
                f"[error] {isomeric_smiles}: "
                f"{result.timing.status} ({result.timing.error_msg})"
            )

    def flush_timings(self) -> None:
        """Serialize the accumulated per-molecule timing records next to the zarr.

        Idempotent: if no records have been collected (e.g. an entirely empty
        build) the file is still emitted as a zero-length JSONL so downstream
        tools can rely on the path existing.
        """
        write_timings_jsonl(
            self._timing_records, self.config.path / "conformer_timings.jsonl"
        )


class FilterMoleculeStage(PipelineStage):
    """SMILES-side filter: ``chemistry.smiles_filter.filter_smiles`` per batch.

    Consumes ``input_batch.raw_smiles`` (raw SMILES strings the generator
    yielded without filtering) and produces ``input_batch.smiles``
    (canonicalized ``SmilesData``). Reindexes every per-system parallel array
    (``regression_data``, ``total_charge``, ``multiplicity``,
    ``structure_ids``) to the surviving rows.

    The internal ``_seen`` set persists across calls, so when a generator
    emits multiple batches in priority order (e.g. a source that emits
    train → valid → test), the first occurrence of a duplicate SMILES wins
    even across batch boundaries.
    """

    def __init__(self, config: SmilesFilterConfig):
        self.config = config
        self._seen: set[str] = set()
        self.load_stats = LoadStats()

    def __call__(
        self, input_batch: InputBatch, data_batch: DataBatch | None = None
    ) -> tuple[InputBatch, DataBatch | None]:
        if input_batch.raw_smiles is None:
            raise ValueError(
                "FilterMoleculeStage requires `raw_smiles` on the InputBatch; "
                "got None. Generators feeding this stage must yield raw "
                "SMILES strings rather than pre-built SmilesData."
            )
        filtered = filter_smiles(input_batch.raw_smiles, self.config, seen=self._seen)
        stats = self.load_stats
        stats.n_raw_rows += len(input_batch.raw_smiles)
        stats.n_invalid_smiles += filtered.invalid
        stats.n_filtered_out += filtered.filtered
        stats.n_duplicates += filtered.duplicates
        stats.n_kept += len(filtered.kept_row_indices)

        _reindex_in_place(input_batch, filtered.kept_row_indices)
        input_batch.smiles = [
            SmilesData(nonisomeric_smiles=nonisomeric, isomeric_smiles=isomeric)
            for isomeric, nonisomeric in zip(
                filtered.isomeric_smiles, filtered.nonisomeric_smiles, strict=True
            )
        ]
        input_batch.raw_smiles = None
        return input_batch, data_batch


class FilterAtomsStage(PipelineStage):
    """Structure-side filter: drop every structure that violates ``GeometryLimits``.

    Operates on ``input_batch.molecules`` (a list of ``ase.Atoms`` straight out
    of an XYZ / SDF / tmQM generator) without re-parsing SMILES.
    """

    def __init__(self, config: GeometryLimits):
        self.config = config
        self._allowed_symbols = config.allowed_element_symbols()
        self.load_stats = LoadStats()

    def __call__(
        self, input_batch: InputBatch, data_batch: DataBatch | None = None
    ) -> tuple[InputBatch, DataBatch | None]:
        if input_batch.molecules is None:
            raise ValueError(
                "FilterAtomsStage requires `molecules` on the InputBatch; got None."
            )
        kept_indices = [
            index
            for index, atoms in enumerate(input_batch.molecules)
            if not geometry_violations(atoms, self.config, self._allowed_symbols)
        ]
        stats = self.load_stats
        stats.n_raw_rows += len(input_batch.molecules)
        stats.n_filtered_out += len(input_batch.molecules) - len(kept_indices)
        stats.n_kept += len(kept_indices)
        _reindex_in_place(input_batch, kept_indices)
        return input_batch, data_batch
