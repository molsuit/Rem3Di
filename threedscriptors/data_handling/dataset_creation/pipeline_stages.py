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

from threedscriptors.configuration.dataset_config import DatasetCreationConfig
from threedscriptors.data_handling.dataset_creation.conformer_timing import (
    ConformerTimingRecord,
    write_timings_jsonl,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    DataBatch,
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from threedscriptors.data_handling.dataset_creation.utils import embed_one_smiles


class PipelineStage(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self, input_batch: InputBatch, data_batch: DataBatch | None = None
    ) -> tuple[InputBatch, DataBatch]:
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
