import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms
from tqdm import tqdm

if TYPE_CHECKING:
    from mace.calculators.mace_torchsim import MaceTorchSimModel

from threedscriptors.configuration.dataset_config import (
    DatasetCreationConfig,
    FilterAtomsStageConfig,
    FilterMoleculeStageConfig,
)
from threedscriptors.data_handling.dataset_creation.build_stats import LoadStats
from threedscriptors.data_handling.dataset_creation.conformer_timing import (
    ConformerTimingRecord,
    write_timings_jsonl,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    apply_smiles_filter,
    resolve_element_set,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    DataBatch,
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from threedscriptors.data_handling.dataset_creation.utils import embed_one_smiles


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
    def __init__(self, dataset_creation_config: DatasetCreationConfig):
        self.config = dataset_creation_config

        self.num_workers = os.cpu_count()
        # Per-stage running totals across all batches; orchestrator reads this
        # at finalize for the build summary.
        from threedscriptors.data_handling.dataset_creation.build_stats import (
            StageStats,
        )

        self.stats = StageStats()
        # Per-molecule timing records, dumped to JSONL by ``flush_timings``.
        self._timing_records: list[ConformerTimingRecord] = []

    def __call__(self, input_batch: InputBatch, data_batch: DataBatch):
        molecules: list[Atoms] = []
        smiles_list: list[SmilesData] = []
        structure_ids: list[StructureID] = []
        parent_idx_for_regression: list[int] = []
        self.stats.n_attempted += len(input_batch.smiles)

        # Submit independent molecules to the pool
        futures = {}
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=None) as ex:
            for mol_i, smi_data in enumerate(input_batch.smiles):
                iso = smi_data.isomeric_smiles
                fut = ex.submit(
                    embed_one_smiles,
                    iso,
                    self.config.N_sampled_conformers,
                    self.config.max_embed_attempts,
                    self.config.max_MMFF_steps,
                    self.config.mmff_non_bonded_thresh,
                )
                futures[fut] = mol_i

            for fut in as_completed(futures):
                mol_i = futures[fut]
                self._collect_one(
                    fut,
                    mol_i=mol_i,
                    input_batch=input_batch,
                    molecules=molecules,
                    smiles_list=smiles_list,
                    structure_ids=structure_ids,
                    parent_idx_for_regression=parent_idx_for_regression,
                )

        input_batch.molecules = molecules
        input_batch.smiles = smiles_list
        input_batch.structure_ids = structure_ids

        if input_batch.total_charge is not None and parent_idx_for_regression:
            input_batch.total_charge = [
                input_batch.total_charge[i] for i in parent_idx_for_regression
            ]
        if input_batch.multiplicity is not None and parent_idx_for_regression:
            input_batch.multiplicity = [
                input_batch.multiplicity[i] for i in parent_idx_for_regression
            ]

        if (
            input_batch.regression_data is not None
            and len(parent_idx_for_regression) > 0
        ):
            idx = np.asarray(parent_idx_for_regression, dtype=np.int64)
            rd = input_batch.regression_data
            new_rd_kwargs: dict = {}
            if rd.targets_system is not None:
                new_rd_kwargs["targets_system"] = rd.targets_system[idx, :]
                new_rd_kwargs["mask_system"] = rd.mask_system[idx, :]
            if rd.targets_atom is not None:
                raise NotImplementedError
            if rd.split is not None:
                new_rd_kwargs["split"] = rd.split[idx]

            input_batch.regression_data = RegressionData(**new_rd_kwargs)

        return input_batch, data_batch

    def _collect_one(
        self,
        fut,
        *,
        mol_i: int,
        input_batch: InputBatch,
        molecules: list[Atoms],
        smiles_list: list[SmilesData],
        structure_ids: list[StructureID],
        parent_idx_for_regression: list[int],
    ) -> None:
        """Drain one worker future, record its timing, and append its confs."""
        isomeric_smiles = input_batch.smiles[mol_i].isomeric_smiles
        nonisomeric_smiles = input_batch.smiles[mol_i].nonisomeric_smiles
        molecule_data_id = input_batch.structure_ids[mol_i].molecule_id
        stereoisomer_id = input_batch.structure_ids[mol_i].stereoisomer_id

        try:
            result = fut.result()
        except Exception as e:
            # Worker process died / pickling error / etc. — synthesize a
            # record so the artifact still accounts for the molecule.
            self.stats.n_other_errors += 1
            self._timing_records.append(
                ConformerTimingRecord(
                    isomeric_smiles=isomeric_smiles,
                    n_atoms=-1,
                    n_confs_requested=int(self.config.N_sampled_conformers),
                    n_confs_emitted=0,
                    t_embed_s=0.0,
                    t_mmff_s=0.0,
                    status="other_error",
                    error_msg=repr(e),
                )
            )
            tqdm.write(f"[error] {isomeric_smiles}: {e!r}")
            return

        self._timing_records.append(result.timing)

        if result.timing.status != "ok":
            # Match the legacy ValueError-vs-other split: parse failures are
            # "value_error"; embed/MMFF failures fall into n_other_errors.
            if result.timing.status == "value_error":
                self.stats.n_value_errors += 1
                tqdm.write(f"[skip] {isomeric_smiles}: {result.timing.error_msg}")
            else:
                self.stats.n_other_errors += 1
                tqdm.write(
                    f"[error] {isomeric_smiles}: "
                    f"{result.timing.status} ({result.timing.error_msg})"
                )
            return

        positions = result.positions
        atomic_numbers = result.atomic_numbers
        assert positions is not None and atomic_numbers is not None
        for k in range(positions.shape[0]):
            molecules.append(
                Atoms(
                    positions=positions[k],
                    numbers=atomic_numbers,
                    pbc=[0, 0, 0],
                )
            )
            smiles_list.append(
                SmilesData(
                    isomeric_smiles=isomeric_smiles,
                    nonisomeric_smiles=nonisomeric_smiles,
                )
            )
            structure_ids.append(
                StructureID(
                    structure_id=-1,
                    molecule_id=molecule_data_id,
                    stereoisomer_id=stereoisomer_id,
                )
            )
            parent_idx_for_regression.append(mol_i)
        self.stats.n_emitted += 1

    def flush_timings(self) -> None:
        """Serialize the accumulated per-mol timing records next to the zarr.

        Idempotent: if no records have been collected (e.g. an entirely empty
        build) the file is still emitted as a zero-length JSONL so downstream
        tools can rely on the path existing.
        """
        out_path = self.config.path / "conformer_timings.jsonl"
        write_timings_jsonl(self._timing_records, out_path)


class ParallelRelaxStage(PipelineStage):
    def __init__(self, mace_model: "MaceTorchSimModel", device, dtype, N_steps: int):
        self.mace_model = mace_model

        self._device = device
        self._dtype = dtype

        self.N_steps = N_steps

    def __call__(self, input_batch, data_batch):
        import torch_sim as ts
        from torch_sim.optimizers import fire

        state = ts.initialize_state(
            input_batch.molecules, device=self._device, dtype=self._dtype
        )

        init_fn, update_fn = fire(
            model=self.mace_model,
        )

        state = init_fn(state)

        for _step in range(self.N_steps):
            state = update_fn(state)

        print(f"Final max force: {torch.linalg.norm(state.forces, dim=1)} eV")

        input_batch.molecules = ts.io.state_to_atoms(state)

        return input_batch, data_batch


class EnantiomaiPairConformalSamplingStage(PipelineStage):
    pass


class MolecularDynamicsConformalSampling(PipelineStage):
    # Sample Conformers from MD simulation
    pass


class FilterMoleculeStage(PipelineStage):
    """SMILES-side filter: parse → standardize → filter → canonicalize → dedupe.

    Consumes ``input_batch.raw_smiles`` (raw SMILES strings the generator
    yielded without filtering) and produces ``input_batch.smiles``
    (canonicalized ``SmilesData``). Reindexes every per-system parallel array
    (``regression_data``, ``total_charge``, ``multiplicity``,
    ``structure_ids``) to the surviving rows.

    The internal ``_seen`` set persists across calls, so when a generator
    emits multiple batches in priority order (e.g. ``TdcGenerator`` emits
    train → valid → test), the first occurrence of a duplicate SMILES wins
    even across batch boundaries.
    """

    def __init__(self, config: FilterMoleculeStageConfig):
        self.config = config
        self.allowed_elements = resolve_element_set(config.element_set)
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

        cfg = self.config
        kept_smiles, kept_idx = apply_smiles_filter(
            input_batch.raw_smiles,
            max_atoms=cfg.max_atoms,
            allowed_elements=self.allowed_elements,
            allow_charged=cfg.allow_charged,
            allow_radicals=cfg.allow_radicals,
            allow_isotopes=cfg.allow_isotopes,
            allow_multifragment=cfg.allow_multifragment,
            strip_salts=cfg.strip_salts,
            neutralize=cfg.neutralize,
            dedupe=cfg.dedupe,
            seen=self._seen,
            stats=self.load_stats,
        )

        _reindex_in_place(input_batch, kept_idx)
        input_batch.smiles = kept_smiles
        input_batch.raw_smiles = None
        return input_batch, data_batch


class FilterAtomsStage(PipelineStage):
    """Atoms-side filter: enforce size, hydrogen-coverage, element-set gates.

    Operates on ``input_batch.molecules`` (a list of ``ase.Atoms`` straight
    out of an XYZ / SDF / tmQM generator) without re-parsing SMILES. Element
    coverage is checked against atomic numbers; SDF-style ``require_3D`` is
    implicit because Atoms only exist when coordinates do.
    """

    def __init__(self, config: FilterAtomsStageConfig):
        self.config = config
        self._allowed_numbers: set[int] | None = None
        if config.element_set is not None:
            from rdkit import Chem as _Chem

            pt = _Chem.GetPeriodicTable()
            self._allowed_numbers = {
                pt.GetAtomicNumber(s) for s in resolve_element_set(config.element_set)
            }
        self.load_stats = LoadStats()

    def __call__(
        self, input_batch: InputBatch, data_batch: DataBatch | None = None
    ) -> tuple[InputBatch, DataBatch | None]:
        if input_batch.molecules is None:
            raise ValueError(
                "FilterAtomsStage requires `molecules` on the InputBatch; "
                "got None."
            )

        cfg = self.config
        stats = self.load_stats
        kept_idx: list[int] = []

        for i, atoms in enumerate(input_batch.molecules):
            stats.n_raw_rows += 1
            n_total = len(atoms)
            if cfg.max_atoms is not None and n_total > cfg.max_atoms:
                stats.n_filtered_out += 1
                continue
            nums = atoms.get_atomic_numbers()
            n_h = int((nums == 1).sum())
            n_heavy = int((nums > 1).sum())
            if n_heavy == 0:
                stats.n_filtered_out += 1
                continue
            if cfg.reject_zero_h and n_h == 0:
                stats.n_filtered_out += 1
                continue
            if cfg.min_h_heavy_ratio > 0.0 and (n_h / n_heavy) < cfg.min_h_heavy_ratio:
                stats.n_filtered_out += 1
                continue
            if self._allowed_numbers is not None and not all(
                int(z) in self._allowed_numbers for z in nums
            ):
                stats.n_filtered_out += 1
                continue
            kept_idx.append(i)

        stats.n_kept += len(kept_idx)
        _reindex_in_place(input_batch, kept_idx)
        return input_batch, data_batch
