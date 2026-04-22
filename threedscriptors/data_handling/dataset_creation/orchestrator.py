from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_creation import (
    DataBatch,
    PipelineStage,
)
from threedscriptors.data_handling.dataset_creation.generators import MoleculeGenerator
from threedscriptors.data_handling.dataset_creation.loading_batch import SmilesData
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from threedscriptors.data_handling.dataset_creation.utils import (
    ensure_numpy_array,
    system_idx_to_ragged_ptr,
)


class DatasetConstructionOrchestrator:
    def __init__(
        self,
        pipeline: Sequence[PipelineStage],
        batch_generator: MoleculeGenerator,
        construction_config: DatasetCreationConfig,
        dataset_config: DatasetConfig,
    ):
        self.pipeline = pipeline
        self.batch_generator = batch_generator
        self.construction_config = construction_config

        # Initialize empty zarr dataset
        self.dataset = MoleculeDataset.create_empty_dataset(
            self.construction_config.path, dataset_config
        )

        # Timing accumulators
        self._stage_times: dict[str, float] = {}
        self._append_time: float = 0.0
        self._num_batches: int = 0

    def build_dataset(self):
        for input_batch in self.batch_generator:
            if input_batch.molecules == [] and input_batch.smiles == []:
                continue

            output_data = None

            for stage in self.pipeline:
                t0 = perf_counter()
                input_batch, output_data = stage(input_batch, output_data)
                dt = perf_counter() - t0
                name = stage.__class__.__name__
                self._stage_times[name] = self._stage_times.get(name, 0.0) + dt

            t0 = perf_counter()

            # if torch.isnan(output_data.embeddings).any():
            #    breakpoint()

            self.append_batch_to_dataset(output_data)
            self._append_time += perf_counter() - t0

            self._num_batches += 1

            if (
                self.construction_config.N_structures is not None
                and self.dataset.N_structures > self.construction_config.N_structures
            ):
                break

        self.finalize()

    def append_batch_to_dataset(self, output_data: DataBatch):
        positions = ensure_numpy_array(output_data.atomic_positions)
        atomic_numbers = ensure_numpy_array(output_data.atomic_numbers)
        total_charge = ensure_numpy_array(output_data.total_charge)
        total_spin = ensure_numpy_array(output_data.total_spin)

        # Ensure pointer length matches the number of structures in the batch.
        ptr = ensure_numpy_array(
            system_idx_to_ragged_ptr(
                output_data.systems_index,
            )
        )

        N_atoms_batch = positions.shape[0]

        if positions.shape != (N_atoms_batch, 3):
            raise ValueError("atomic_positions must be [N_atoms, 3]")
        if atomic_numbers.shape[0] != N_atoms_batch:
            raise ValueError("atomic_numbers must have length N_atoms")
        if output_data.systems_index.shape[0] != N_atoms_batch:
            raise ValueError("systems_index must have length N_atoms")

        molecule_ids, stereoisomer_ids = self.get_mol_ids_for_batch(
            output_data.smiles_data, output_data.structure_ids
        )

        if output_data.regression_data is not None:
            system_targets = output_data.regression_data.targets_system
            system_masks = output_data.regression_data.mask_system
            atom_target = output_data.regression_data.targets_atom
            atom_mask = output_data.regression_data.mask_atom
        else:
            system_targets = None
            system_masks = None
            atom_target = None
            atom_mask = None

        self.dataset.append_batch(
            positions,
            atomic_numbers,
            ptr,
            molecule_ids,
            stereoisomer_ids,
            total_charge,
            total_spin,
            system_targets,
            system_masks,
            atom_target,
            atom_mask,
        )

    def get_mol_ids_for_batch(
        self, smiles_data: list[SmilesData], structure_ids: list[StructureID]
    ):
        if self.dataset.config.contains_smiles:
            assert len(smiles_data) == len(structure_ids)

            new_smiles = [sd.nonisomeric_smiles for sd in smiles_data]
            new_smiles_to_id_map = self.dataset.smiles.append_new_lines(new_smiles)

            molecule_ids = np.array([new_smiles_to_id_map[s] for s in new_smiles])

            new_isomeric_smiles = [sd.isomeric_smiles for sd in smiles_data]
            new_isomeric_smiles_to_id_map = (
                self.dataset.isomeric_smiles.append_new_lines(new_isomeric_smiles)
            )
            stereoisomer_ids = np.array(
                [new_isomeric_smiles_to_id_map[s] for s in new_isomeric_smiles]
            )

        else:
            molecule_ids = np.array([sid.molecule_id for sid in structure_ids])
            stereoisomer_ids = np.array([sid.stereoisomer_id for sid in structure_ids])

        return molecule_ids, stereoisomer_ids

    def finalize(self):
        # Close all the files, ensure that everything is stored correctly
        if self.dataset.config.contains_smiles:
            self.dataset.smiles.close()
            self.dataset.isomeric_smiles.close()

        self.dataset.shrink_to_fit()

        # Print simple timing summary
        if self._num_batches > 0:
            per_batch_append = self._append_time / self._num_batches
            # Order stages by total time (descending)
            ordered = sorted(
                self._stage_times.items(), key=lambda x: x[1], reverse=True
            )
            print("Dataset construction timing summary:")
            for name, total in ordered:
                print(
                    f"  Stage {name}: {total:.3f}s total ({total / self._num_batches:.4f}s/batch)"
                )
            print(
                f"  Append: {self._append_time:.3f}s total ({per_batch_append:.4f}s/batch)"
            )
