from collections.abc import Sequence
from typing import List

from threedscriptors.data_handling.mol_id import StructureID

import torch

from threedscriptors.configuration.data_config import DatasetConfig, DatasetSplit
from threedscriptors.data_handling.dataset import (
    AtomicEmbeddingWithPositionsDataset,
    BaseDataset,
)
from threedscriptors.data_handling.dataset_builder import DatasetBuilder


class DatasetConcatenation:
    def __init__(self, datasets: Sequence[BaseDataset]):
        # takes in a list of datasets
        self.datasets = datasets

        new_dataset_config = self.get_new_dataset_config()

        self.new_dataset = new_dataset_config.dataset_type.value(
            dataset_config=new_dataset_config
        )

    def get_new_dataset_config(self):
        embedding_models = set(
            [d.dataset_config.embedding_model_config.model_name for d in self.datasets]
        )
        assert len(embedding_models) == 1

        heavy_atoms_only = [d.dataset_config.only_heavy_atoms for d in self.datasets]

        assert all([heavy_atoms_only[0] == h for h in heavy_atoms_only])

        dataset_split = DatasetSplit.TRAIN

        new_dataset_config = DatasetConfig(
            N_molecules=sum([d.dataset_config.N_molecules for d in self.datasets]),
            dataset_type=AtomicEmbeddingWithPositionsDataset,
            BFGS_tol=max([d.dataset_config.BFGS_tol for d in self.datasets]),
            BFGS_max_steps=max(
                [d.dataset_config.BFGS_max_steps for d in self.datasets]
            ),
            N_conformers=max(
                [d.dataset_config.N_conformers for d in self.datasets]
            ),  # This does not really make sense. How should we treat different datasets with varying_ N_conformers?
            max_atoms=max([d.dataset_config.max_atoms for d in self.datasets]),
            embedding_model_config=self.datasets[
                0
            ].dataset_config.embedding_model_config,
            tasks= None, #[task for d in self.datasets for task in d.dataset_config.tasks],
            dataset_name="+".join(
                [d.dataset_config.dataset_name for d in self.datasets]
            ),
            dataset_split=dataset_split,
            only_heavy_atoms=heavy_atoms_only[0],
        )

        task_names = None #[task.task_name for task in new_dataset_config.tasks]
        # Append the dataset configs. assert no tasks have the same name

        #assert len(task_names) == len(set(task_names)), "Found duplicate task_names"

        return new_dataset_config

    def concatenate(self):
        self.concatenate_molecules()
        self.concatenate_atomic_embeddings()
        #self.concatenate_regression_targets()
        #self.concatenate_atomic_positions()
        # self.concatenate_auxillary_data()
        # self.concatenate_structural_encodings()

        return self.new_dataset


    


    


    def concatenate_molecules(self):
        # Adds mol_ids and smiles

        structure_id_start = 0
        molecule_id_start = 0

        new_structure_ids : list[StructureID] = []

        for d in self.datasets:

            dataset_builder = DatasetBuilder(d)

            new_structure_ids.extend(
                dataset_builder.canonicalize_structure_ids(
                    structure_id_start, molecule_id_start
                )
            )
            structure_id_start = len(new_structure_ids)
            molecule_id_start = new_structure_ids[-1].molecule_id + 1


        self.new_dataset.structure_ids = new_structure_ids

        if all([d.smiles_list is not None for d in self.datasets]):
            new_smiles_list = [smi for d in self.datasets for smi in d.smiles_list]
            self.new_dataset.smiles_list = new_smiles_list
        else:
            raise ValueError

        if all([d.molecules is not None for d in self.datasets]):
            new_molecules = [mol for d in self.datasets for mol in d.molecules]
            self.new_dataset.molecules = new_molecules
        else: 
            raise ValueError

    def recanonicalize_mol_ids(self, mol_ids=list[list[int]]):
        mol_ids_sets = [set(ids) for ids in mol_ids]

        running_shift = 0

        self.collected_relabel_dicts: list[dict[int:int]] = []

        for dataset_mol_ids in mol_ids_sets:
            current_relabel_dict = {
                old: new for new, old in enumerate(dataset_mol_ids, start=running_shift)
            }

            running_shift = max(current_relabel_dict.values()) + 1
            self.collected_relabel_dicts.append(current_relabel_dict)

        new_mol_ids = []

        for original_mol_ids, relabel_dict in zip(
            mol_ids, self.collected_relabel_dicts, strict=False
        ):
            new_mol_ids.extend([relabel_dict[m] for m in original_mol_ids])

        return new_mol_ids

    def concatenate_atomic_embeddings(self):
        # expand the padding mask and atomic embdding mask to max dimension

        for dataset in self.datasets:
            if (
                dataset.dataset_config.max_atoms
                != self.new_dataset.dataset_config.max_atoms
            ):
                dataset.expand_embedding_num_atoms(
                    new_max_num_atoms=self.new_dataset.dataset_config.max_atoms
                )

        new_embeddings = torch.cat([d.embeddings for d in self.datasets])
        new_padding_masks = torch.cat([d.padding_mask for d in self.datasets])

        new_pos = [d.atomic_positions for d in self.datasets]
        new_atomic_positions = torch.cat(new_pos)

        self.new_dataset.embeddings = new_embeddings
        self.new_dataset.padding_mask = new_padding_masks
        self.new_dataset.atomic_positions = new_atomic_positions

    def concatenate_regression_targets(self):
        # Creates the Block matrices of regression targets, and regression masks.

        collected_regression_masks = [d.regression_masks for d in self.datasets]
        collected_regression_targets = [d.regression_targets for d in self.datasets]

        new_regression_targets = torch.block_diag(*collected_regression_targets)
        new_regression_masks = torch.block_diag(*collected_regression_masks)
        self.new_dataset.regression_targets = new_regression_targets
        self.new_dataset.regression_masks = new_regression_masks

    def concatenate_auxillary_data(self):
        auxillary_data_keys = {}

        # Collects all present auxillary data keys and their dimensions
        for d in self.datasets:
            if d.auxillary_data is not None:
                for k, v in d.auxillary_data.items():
                    auxillary_data_keys[k] = v.shape

        # if there is non aux data at all, then leave fn
        if auxillary_data_keys == {}:
            return

        # Construct the padding aux data in each task
        for d in self.datasets:
            if d.auxillary_data is None:
                d.auxillary_data = {}

            for k, torch_sizes in auxillary_data_keys.items():
                if k not in d.auxillary_data:
                    d.auxillary_data[k] = torch.zeros(torch_sizes, dtype=torch.float32)

        new_aux_data = {}
        # Construct the completely merged aux data dict
        for key in auxillary_data_keys.keys():
            new_values = torch.cat([d.auxillary_data[key] for d in self.datasets])
            new_aux_data[key] = new_values.float()

        self.new_dataset.auxillary_data = new_aux_data

    def concatenate_structural_encodings(self):

        collected_transition_matrices = [
            d.random_walk_transition_matrix for d in self.datasets
        ]

        self.new_dataset.random_walk_transition_matrix = torch.cat(
            collected_transition_matrices
        )
