import os
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch import from_numpy
from tqdm import tqdm

from threedprints.data_handling.data_config import DatasetConfig
from threedprints.data_handling.preprocessing import get_ase_atoms, get_mace_descriptors
from threedprints.data_handling.smiles_iterator import SmilesIterator
from threedprints.utils.config_utils import from_yaml, to_yaml


class BaseAtomicDataset(data.Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def store_data_to_disk(self, directory):
        pass


class AtomEmbeddingDataset(BaseAtomicDataset):
    def __init__(
        self, embeddings, padding_mask, dataset_config: DatasetConfig, smiles_list
    ):
        super().__init__()
        self.embeddings = embeddings
        self.padding_masks = padding_mask
        self.dataset_config = dataset_config
        self.smiles_list = smiles_list

    def __len__(self):
        return self.padding_masks.shape[0]

    def __getitem__(self, index):
        embeddings = self.embeddings[index]
        padding_mask = self.padding_masks[index]

        return embeddings, padding_mask

    def store_data_to_disk(self, directory):
        os.makedirs(directory, exist_ok=True)
        # Save embeddings/dataset_config to disk
        np.save(f"{directory}/descriptor_data.npy", self.embeddings)
        np.save(f"{directory}/descriptor_mask.npy", self.padding_masks)

        with open(f"{directory}/smiles_list", "w") as f:
            for i in self.smiles_list:
                f.write(i + "\n")

        to_yaml(f"{directory}/dataset_config.yaml", self.dataset_config)

    def extend_max_atoms(self, new_max_atoms):
        raise NotImplementedError

        self.metadata["max_atoms"] = new_max_atoms
        self.embeddings = np.pad(
            self.embeddings,
            ((0, 0), (0, new_max_atoms - self.embeddings.shape[1]), (0, 0)),
        )
        self.padding_masks = np.pad(
            self.padding_masks,
            ((0, 0), (0, new_max_atoms - self.padding_masks.shape[1])),
        )


class RegressionAtomEmbeddingDataset(BaseAtomicDataset):
    def __init__(
        self,
        embeddings,
        padding_mask,
        regression_targets,
        regression_masks,
        dataset_config: DatasetConfig,
        smiles_list,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.padding_masks = padding_mask
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks
        self.dataset_config = dataset_config
        self.smiles_list = smiles_list

    def __len__(self):
        return self.padding_masks.shape[0]

    def __getitem__(self, index):
        embeddings = self.embeddings[index]
        padding_mask = self.padding_masks[index]
        regression_targets = self.regression_targets[index]
        regression_masks = self.regression_masks[index]

        return embeddings, padding_mask, regression_targets, regression_masks

    def store_data_to_disk(self, directory):
        # ensure that directory exists:
        os.makedirs(directory, exist_ok=True)

        # Save embeddings/dataset_config to disk
        np.save(f"{directory}/descriptor_data.npy", self.embeddings)
        np.save(f"{directory}/descriptor_mask.npy", self.padding_masks)
        np.save(f"{directory}/regression_targets.npy", self.regression_targets)
        np.save(f"{directory}/regression_masks.npy", self.regression_masks)

        with open(f"{directory}/smiles_list", "w") as f:
            for i in self.smiles_list:
                f.write(i + "\n")

        to_yaml(f"{directory}/dataset_config.yaml", self.dataset_config)

    def normalize_targets(self, mean=None, std=None):
        # Normalize the regression targets
        if mean is None:
            mean = torch.mean(a=self.regression_targets, axis=0)
        if std is None:
            std = torch.std(self.regression_targets, axis=0)

        self.regression_targets = (self.regression_targets - mean) / std

        self.dataset_config.mean = mean
        self.dataset_config.std = std


class DatasetFactory:
    @classmethod
    def from_disk(cls, directory: str):
        # Load dataset_config first
        dataset_config: DatasetConfig = from_yaml(
            f"{directory}/dataset_config.yaml", DatasetConfig
        )

        embeddings_arr = np.load(directory + "/descriptor_data.npy")
        padding_mask_arr = np.load(directory + "/descriptor_mask.npy")

        embeddings = from_numpy(embeddings_arr).float()
        padding_mask = from_numpy(padding_mask_arr).float()

        with open(directory + "/smiles_list") as f:
            smiles_list = f.readlines()
            smiles_list = [i.strip() for i in smiles_list]

        if dataset_config.dataset_type == "Pretraining":
            return AtomEmbeddingDataset(
                embeddings, padding_mask, dataset_config, smiles_list
            )

        elif dataset_config.dataset_type == "Regression":
            regression_targets_arr = np.load(directory + "/regression_targets.npy")
            regression_masks_arr = np.load(directory + "/regression_masks.npy")

            regression_targets = from_numpy(regression_targets_arr).float()
            regression_masks = from_numpy(regression_masks_arr).float()

            return RegressionAtomEmbeddingDataset(
                embeddings,
                padding_mask,
                regression_targets,
                regression_masks,
                dataset_config,
                smiles_list,
            )

        else:
            raise ValueError("Unknown dataset type")

    @classmethod
    def from_smiles(
        cls,
        iterator: SmilesIterator,
        mace_caluclator,
        dataset_config: DatasetConfig,
        regression_target: np.ndarray | None = None,
        regression_masks: np.ndarray | None = None,
    ):
        assert dataset_config.dataset_type in [
            "Pretraining",
            "Regression",
            "Evaluation",
        ]

        if dataset_config.dataset_type == "Regression":
            assert regression_target is not None
            assert regression_masks is not None

        smiles_list = []
        index_list = []

        embeddings = np.zeros(
            (
                dataset_config.N_molecules,
                dataset_config.max_atoms,
                dataset_config.embedding_size,
            )
        )
        padding_mask = np.zeros((dataset_config.N_molecules, dataset_config.max_atoms))

        i = 0

        with tqdm(total=dataset_config.N_molecules) as pbar:
            while i < dataset_config.N_molecules:
                try:
                    smiles = next(iterator)
                except StopIteration:
                    print(
                        f"Reached StopIteration prematurely. Completed {i} reading molecules."
                    )
                    break

                try:
                    atoms = get_ase_atoms(smiles)
                    # Sample conformers?
                    descriptor = get_mace_descriptors(
                        atoms,
                        mace_caluclator,
                        dataset_config.BFGS_tol,
                        max_steps=dataset_config.BFGS_max_steps,
                    )

                except ValueError as ve:
                    tqdm.write(f"Error with Smiles {smiles}: {ve}")
                    continue

                else:
                    smiles_list.append(smiles)
                    index_list.append(i)
                    embeddings[i, 0 : descriptor.shape[0], :] = descriptor
                    padding_mask[i, 0 : descriptor.shape[0]] = 1
                    i = i + 1
                    pbar.update(1)

        # Check whether the dataset actually contains the desired number of molecules ???
        num_molecules = len(smiles_list)
        dataset_config.N_molecules = num_molecules
        embeddings = embeddings[:num_molecules]
        padding_mask = padding_mask[:num_molecules]

        if (
            dataset_config.dataset_type == "Pretraining"
            or dataset_config.dataset_type == "Evaluation"
        ):
            return AtomEmbeddingDataset(
                embeddings, padding_mask, dataset_config, smiles_list
            )
        elif dataset_config.dataset_type == "Regression":
            # get the regression targets for the molecules
            regression_target = regression_target[index_list, :]
            regression_masks = regression_masks[index_list, :]

            return RegressionAtomEmbeddingDataset(
                embeddings,
                padding_mask,
                regression_target,
                regression_masks,
                dataset_config,
                smiles_list,
            )

    def from_smiles_and_positions():
        # Inialize descriptor from already relaxed positions which forgoes expensive relaxation with MACE
        # Maybe check forces of DFT relaxed structures with MACE
        raise NotImplementedError

    @classmethod
    def from_smiles_with_hdf5_caching(
        cls,
        iterator: SmilesIterator,
        mace_caluclator,
        metadata,
        regression_target: np.ndarray | None = None,
        regression_masks: np.ndarray | None = None,
        chunk_size=500,
        directory=None,
    ):
        # Load the data from smiles, while intermittently storing the data to disk, to avoid memory issues with very large datasets

        # Store N_molecules temporarily.
        raise NotImplementedError
        with h5py.File(f"{directory}/dataset.hdf5", "w") as f:
            # Create datasets for the embeddings, padding masks, regression targets and regression masks
            embeddings = f.create_dataset(
                "embeddings",
                shape=(
                    metadata["N_molecules"],
                    metadata["max_atoms"],
                    metadata["embedding_size"],
                ),
                dtype="f",
            )
            padding_masks = f.create_dataset(
                "padding_masks",
                shape=(metadata["N_molecules"], metadata["max_atoms"]),
                dtype=np.bool_,
            )

            total_N_molecules = metadata["N_molecules"]
            metadata["N_molecules"] = chunk_size

            i = 0
            while i < total_N_molecules:
                # Call the from smiles method on each chunk
                dataset = cls.from_smiles(iterator, mace_caluclator, metadata)
                # Check the total size of the returned dataset
                returned_molecules = len(dataset)
                # Add the resulting dataset to the hdf5 file
                embeddings[i : i + returned_molecules, :, :] = dataset.embeddings
                padding_masks[i : i + returned_molecules, :] = dataset.padding_masks

                # Increase the starting point
                i = i + chunk_size
