import os

import numpy as np
import torch
import torch.utils.data as data
from ase import Atoms

from threedscriptors.configuration.config_utils import to_yaml
from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.data_utils import (
    get_max_molecule_size,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator

type Molecules = list[Atoms]


class BaseAtomicDataset(data.Dataset):
    def __init__(
        self,
        molecules: Molecules,
        dataset_config: DatasetConfig,
        smiles_list: list[str],
    ):
        super().__init__()
        self.molecules: Molecules = molecules
        self.dataset_config: DatasetConfig = dataset_config
        self.smiles_list: list[str] = smiles_list

        # Create all the fields for the child classes, which allows us to unify the store data to disk classes

        self.embeddings = None
        self.padding_mask = None
        self.regression_targets = None
        self.regression_masks = None
        self.auxillary_data = None

    def __len__(self):
        return len(self.molecules)

    def get_max_atoms(self):
        if self.dataset_config.max_atoms is None:
            smiles_iterator = ListSmilesIterator(self.smiles_list)
            self.dataset_config.max_atoms = get_max_molecule_size(smiles_iterator)

        return self.dataset_config.max_atoms

    def expand_embedding_num_atoms(self, new_max_num_atoms):
        # Method can be used to increase the "Sequence length" i.e the number of atoms in a molecule. So that the embeddings do not have to be recalculated.
        raise NotImplementedError

    def store_data_to_disk(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        # Store Positions
        padding_dim, padded_positions, padded_atomic_numbers = (
            self.get_padded_positions()
        )

        np.save(f"{directory}/padded_positions.npy", padded_positions)
        np.save(f"{directory}/padding_dim.npy", padding_dim)
        np.save(f"{directory}/padded_atomic_numbers.npy", padded_atomic_numbers)

        # Store Embeddings
        if self.embeddings is not None:
            np.save(f"{directory}/embeddings.npy", self.embeddings)
            np.save(f"{directory}/padding_mask.npy", self.padding_mask)

        # Store Regression Targets
        if self.regression_targets is not None:
            if self.dataset_config.is_normalized:
                # undo the normalization
                raise ValueError

            np.save(f"{directory}/regression_targets.npy", self.regression_targets)
            np.save(f"{directory}/regression_masks.npy", self.regression_masks)

        # Store Auxillary Data
        if self.auxillary_data is not None:
            np.savez(f"{directory}/auxillary_data.npz", **self.auxillary_data)

        # Store Smiles
        with open(f"{directory}/smiles_list", "w") as f:
            for i in self.smiles_list:
                f.write(i + "\n")

        # Store Config
        to_yaml(f"{directory}/dataset_config.yaml", self.dataset_config)

    def get_padded_positions(self):
        positions = [at.get_positions() for at in self.molecules]
        atomic_numbers = [at.get_atomic_numbers() for at in self.molecules]
        padding_dim = np.array([len(an) for an in atomic_numbers])

        padded_atomic_numbers = np.array(
            [
                np.pad(
                    an, (0, self.dataset_config.max_atoms - len(an)), mode="constant"
                )
                for an in atomic_numbers
            ]
        )

        # For positions, assuming each position array has shape (n_atoms, 3)
        padded_positions = np.array(
            [
                np.pad(
                    pos,
                    ((0, self.dataset_config.max_atoms - pos.shape[0]), (0, 0)),
                    mode="constant",
                )
                for pos in positions
            ]
        )

        return padding_dim, padded_positions, padded_atomic_numbers

    def normalize_regression_targets(self):
        assert self.regression_targets is not None

        if self.dataset_config.is_normalized:
            print("Dataset was already normalized")
            return

        regression_targets = self.regression_targets
        if isinstance(self.regression_targets, torch.Tensor):
            regression_targets = self.regression_targets.detach().cpu().numpy()

        mean = np.mean(regression_targets, axis=0, where=self.regression_masks)

        std = np.std(regression_targets, axis=0, where=self.regression_masks)

        self.regression_targets = (regression_targets - mean) / std

        for task, task_mean, task_std in zip(
            self.dataset_config.tasks, mean.tolist(), std.tolist(), strict=False
        ):
            task.mean = task_mean
            task.std = task_std

        self.dataset_config.is_normalized = True

        self.regression_targets = torch.Tensor(self.regression_targets)

    def undo_regression_target_normalization(self):
        assert self.regression_targets is not None
        if not self.dataset_config.is_normalized:
            print("Dataset was already unnormalized")

        raise NotImplementedError


class AtomEmbeddingDataset(BaseAtomicDataset):
    def __init__(
        self,
        molecules: Molecules,
        dataset_config: DatasetConfig,
        smiles_list: list[str],
        embeddings: np.ndarray,
        padding_mask: np.ndarray,
    ):
        super().__init__(molecules, dataset_config, smiles_list)

        self.embeddings = embeddings
        self.padding_mask = padding_mask

    def __getitem__(self, index):
        return self.embeddings[index], self.padding_mask[index]


class RegressionAtomEmbeddingDataset(BaseAtomicDataset):
    def __init__(
        self,
        molecules: Molecules,
        dataset_config: DatasetConfig,
        smiles_list: list[str],
        embeddings: np.ndarray,
        padding_mask: np.ndarray,
        regression_targets: np.ndarray,
        regression_masks: np.ndarray,
    ):
        super().__init__(
            molecules=molecules, dataset_config=dataset_config, smiles_list=smiles_list
        )

        self.embeddings = embeddings
        self.padding_mask = padding_mask
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks

    def __getitem__(self, index):
        embeddings = self.embeddings[index]
        padding_mask = self.padding_mask[index]

        regression_targets = self.regression_targets[index]
        regression_masks = self.regression_masks[index]

        return embeddings, padding_mask, regression_targets, regression_masks


class RegressionAtomEmbeddingDatasetWithAuxillaryData(RegressionAtomEmbeddingDataset):
    # ^^Maybe there is a better name for this??? :D

    def __init__(
        self,
        molecules: Molecules,
        dataset_config: DatasetConfig,
        smiles_list: list[str],
        embeddings: np.ndarray,
        padding_mask: np.ndarray,
        regression_targets: np.ndarray,
        regression_masks: np.ndarray,
        auxillary_data: dict[str : np.ndarray],
    ):
        super().__init__(
            molecules=molecules,
            dataset_config=dataset_config,
            smiles_list=smiles_list,
            embeddings=embeddings,
            padding_mask=padding_mask,
            regression_targets=regression_targets,
            regression_masks=regression_masks,
        )

        self.auxillary_data = auxillary_data

    def __getitem__(self, index):
        embeddings = self.embeddings[index]
        padding_mask = self.padding_mask[index]

        regression_targets = self.regression_targets[index]
        regression_masks = self.regression_masks[index]

        auxillary_data = {
            k: self.auxillary_data[k][index, :] for k in self.auxillary_data.keys()
        }

        return (
            embeddings,
            padding_mask,
            regression_targets,
            regression_masks,
            auxillary_data,
        )

    @staticmethod
    def custom_auxillary_data_collate_fn(batch):
        raise NotImplementedError
