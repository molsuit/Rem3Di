from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import torch
import torch_sim as ts
from ase import Atoms
from mace.calculators import MACECalculator
from torch_sim.models.mace import MaceModel
from tqdm import tqdm

from threedscriptors.configuration.dataset_config import DatasetCreationConfig
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    DataBatch,
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from threedscriptors.data_handling.dataset_creation.utils import embed_one_smiles
from torch_sim.optimizers import fire

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


class BatchedEmbeddingStage(PipelineStage):

    def __init__(self, mace_model: MaceModel, device, dtype):
        self.mace_model = mace_model

        self._device = device
        self._dtype = dtype

    def __call__(self, input_batch, data_batch):

        state = ts.initialize_state(
            input_batch.molecules, device=self._device, dtype=self._dtype
        )
        out = self.mace_model(state)

        if data_batch is not None:
            # Databatch has already been initialized
            data_batch.atomic_positions = state.positions
            data_batch.atomic_numbers = state.atomic_numbers
            data_batch.embeddings = out["descriptors"]
            data_batch.systems_index = state.system_idx

        else:
            data_batch = DataBatch(
                atomic_positions=state.positions,
                atomic_numbers=state.atomic_numbers,
                embeddings=out["descriptors"],
                systems_index=state.system_idx,
                smiles_data=input_batch.smiles,
                structure_ids=input_batch.structure_ids,
            )

        return input_batch, data_batch


class SequentialEmbeddingStage(PipelineStage):

    def __init__(self, mace_calculator: MACECalculator, device, dtype):

        self.mace_calc = mace_calculator
        self._device = device
        self._dtype = dtype

    def __call__(self, input_batch, data_batch):

        embeddings = []
        positions = []
        atomic_numbers = []
        systems_index = []

        for i, atoms in enumerate(input_batch.molecules):
            des = self.mace_calc.get_descriptors(atoms, invariants_only=False)

            embeddings.append(torch.from_numpy(des))
            positions.append(torch.as_tensor(atoms.get_positions(), dtype=self._dtype))
            atomic_numbers.append(
                torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            )
            systems_index.append(torch.full((len(atoms),), i, dtype=torch.long))

        if data_batch is not None:
            data_batch.embeddings = torch.cat(embeddings, dim=0)
            data_batch.atomic_positions = torch.cat(positions, dim=0)
            data_batch.atomic_numbers = torch.cat(atomic_numbers)
            data_batch.systems_index = torch.cat(systems_index)

        else:
            data_batch = DataBatch(
                embeddings=torch.cat(embeddings, dim=0),
                atomic_positions=torch.cat(positions, dim=0),
                atomic_numbers=torch.cat(atomic_numbers),
                systems_index=torch.cat(systems_index),
                smiles_data=input_batch.smiles,
                structure_ids=input_batch.structure_ids,
            )
        return input_batch, data_batch


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
                    # Sanity: use the returned noniso if you want; here we trust the original inputs
                    K = positions.shape[0]

                    # Build ASE atoms in the parent process
                    for k in range(K):
                        atoms = Atoms(positions=positions[k], numbers=atomic_numbers, pbc = [0,0,0])
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

        # Overwrite the original smiles and molecules
        input_batch.molecules = molecules
        input_batch.smiles = smiles_list
        input_batch.structure_ids = structure_ids

        # Duplicate regression rows to match generated conformers
        if input_batch.regression_data is not None and len(parent_idx_for_regression) > 0:
            idx = np.asarray(parent_idx_for_regression, dtype=np.int64)
            rd = input_batch.regression_data
            new_targets = rd.regression_targets[idx, :]
            # NOTE: fix the tiny bug — remove the trailing comma so it's not a tuple
            new_masks = rd.regression_masks[idx, :]
            input_batch.regression_data = RegressionData(
                regression_targets=new_targets,
                regression_masks=new_masks,
            )

        return input_batch, data_batch


class ParallelRelaxStage(PipelineStage):

    def __init__(self, mace_model: MaceModel, device, dtype, N_steps: int):
        self.mace_model = mace_model

        self._device = device
        self._dtype = dtype

        self.N_steps = N_steps

    def __call__(self, input_batch, data_batch):

        state = ts.initialize_state(
            input_batch.molecules, device=self._device, dtype=self._dtype
        )

        # Initialize unit cell gradient descent optimizer
        init_fn, update_fn = fire(
            model=self.mace_model,
        )

        state = init_fn(state)

        # Run optimization for a few steps
        
        for step in range(self.N_steps):

            state = update_fn(state)

        print(f"Final max force: {torch.linalg.norm(state.forces, dim = 1)} eV")

        input_batch.molecules = ts.io.state_to_atoms(state)

        return input_batch, data_batch

class RandomWalkTransitionMatrix(PipelineStage):
    pass

    # Implements calculation of the Transition matrix for 2D positional encodings, should be stored in sparse format. 


class EnantiomaiPairConformalSamplingStage(PipelineStage):
    pass


class MolecularDynamicsConformalSampling(PipelineStage):
    # Sample Conformers from MD simulation 
    pass