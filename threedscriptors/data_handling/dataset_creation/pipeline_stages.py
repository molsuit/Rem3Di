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


def _per_system_charge_spin(
    input_batch: InputBatch, n_systems: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_batch.total_charge is None:
        charge = torch.zeros(n_systems, dtype=dtype)
    else:
        charge = torch.as_tensor(input_batch.total_charge, dtype=dtype)
    if input_batch.total_spin is None:
        spin = torch.zeros(n_systems, dtype=dtype)
    else:
        spin = torch.as_tensor(input_batch.total_spin, dtype=dtype)

    if charge.shape[0] != n_systems or spin.shape[0] != n_systems:
        raise ValueError(
            f"total_charge / total_spin must have length {n_systems}, "
            f"got {charge.shape[0]} / {spin.shape[0]}"
        )
    return charge, spin


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
        charge, spin = _per_system_charge_spin(input_batch, n_systems, self._dtype)

        output_batch = DataBatch(
            atomic_positions=positions,
            atomic_numbers=atomic_numbers,
            systems_index=system_idx,
            smiles_data=input_batch.smiles,
            structure_ids=input_batch.structure_ids,
            total_charge=charge,
            total_spin=spin,
            regression_data=input_batch.regression_data,
        )

        return input_batch, output_batch


class ConformerGenerationStage(PipelineStage):
    def __init__(self, dataset_creation_config: DatasetCreationConfig):
        self.config = dataset_creation_config

        self.num_workers = os.cpu_count()

    def __call__(self, input_batch: InputBatch, data_batch: DataBatch):
        molecules: list[Atoms] = []
        smiles_list: list[SmilesData] = []
        structure_ids: list[StructureID] = []
        parent_idx_for_regression: list[int] = []

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
                )
                futures[fut] = mol_i

            for fut in as_completed(futures):
                mol_i = futures[fut]
                isomeric_smiles = input_batch.smiles[mol_i].isomeric_smiles
                nonisomeric_smiles = input_batch.smiles[mol_i].nonisomeric_smiles
                molecule_data_id = input_batch.structure_ids[mol_i].molecule_id
                stereoisomer_id = input_batch.structure_ids[mol_i].stereoisomer_id

                try:
                    iso_smi, noniso_smi, positions, atomic_numbers = fut.result()
                    K = positions.shape[0]

                    for k in range(K):
                        atoms = Atoms(
                            positions=positions[k],
                            numbers=atomic_numbers,
                            pbc=[0, 0, 0],
                        )
                        molecules.append(atoms)
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

                except ValueError as ve:
                    tqdm.write(f"[skip] {isomeric_smiles}: {ve}")
                except Exception as e:
                    tqdm.write(f"[error] {isomeric_smiles}: {e!r}")

        input_batch.molecules = molecules
        input_batch.smiles = smiles_list
        input_batch.structure_ids = structure_ids

        if input_batch.total_charge is not None and parent_idx_for_regression:
            input_batch.total_charge = [
                input_batch.total_charge[i] for i in parent_idx_for_regression
            ]
        if input_batch.total_spin is not None and parent_idx_for_regression:
            input_batch.total_spin = [
                input_batch.total_spin[i] for i in parent_idx_for_regression
            ]

        if (
            input_batch.regression_data is not None
            and len(parent_idx_for_regression) > 0
        ):
            idx = np.asarray(parent_idx_for_regression, dtype=np.int64)
            rd = input_batch.regression_data
            if rd.targets_system is not None:
                new_targets = rd.targets_system[idx, :]
                new_masks = rd.mask_system[idx, :]
            if rd.targets_atom is not None:
                raise NotImplementedError

            input_batch.regression_data = RegressionData(
                targets_system=new_targets,
                mask_system=new_masks,
            )

        return input_batch, data_batch


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
