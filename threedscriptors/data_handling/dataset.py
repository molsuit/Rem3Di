import os
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch.utils.data as data
from ase import Atoms
from mace.calculators import MACECalculator
from torch import from_numpy
from tqdm import tqdm

from threedscriptors.configuration.config_utils import from_yaml, to_yaml
from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.preprocessing import get_relaxed_conformers
from threedscriptors.data_handling.smiles_iterator import SmilesIterator

type Molecules = list[Atoms]


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
        self,
        molecules: Molecules,
        dataset_config: DatasetConfig,
        smiles_list: list[str],
    ):
        super().__init__()
        self.molecules: Molecules = molecules
        self.dataset_config: DatasetConfig = dataset_config
        self.smiles_list: list[str] = smiles_list

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        return self.embeddings[index], self.padding_mask[index]

    def store_data_to_disk(self, directory):
        os.makedirs(directory, exist_ok=True)
        # Save embeddings/dataset_config to disk

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
        np.save(f"{directory}/padded_positions.npy", padded_positions)
        np.save(f"{directory}/padding_dim.npy", padding_dim)
        np.save(f"{directory}/padded_atomic_numbers.npy", padded_atomic_numbers)

        with open(f"{directory}/smiles_list", "w") as f:
            for i in self.smiles_list:
                f.write(i + "\n")

        to_yaml(f"{directory}/dataset_config.yaml", self.dataset_config)

    def calculate_embeddings(self, calculator: MACECalculator, embedding_size: int):
        embeddings = np.zeros(
            shape=(
                self.dataset_config.N_molecules,
                self.dataset_config.max_atoms,
                embedding_size,
            )
        )
        padding_mask = np.ones(
            shape=(self.dataset_config.N_molecules, self.dataset_config.max_atoms)
        )  # Integer 1 = Boolean True = means that this position is padding

        for i, atoms in enumerate(self.molecules):
            descriptors = calculator.get_descriptors(atoms, invariants_only=False)
            num_atoms = len(atoms.get_atomic_numbers())
            embeddings[i, :num_atoms, :] = descriptors
            padding_mask[i, :num_atoms] = 0

        self.embeddings = embeddings
        self.padding_mask = padding_mask


class RegressionAtomEmbeddingDataset(BaseAtomicDataset):
    def __init__(
        self,
        molecules: Molecules,
        regression_targets,
        regression_masks,
        dataset_config: DatasetConfig,
        smiles_list: list[str],
    ):
        super().__init__()
        self.molecules: Molecules = molecules
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks
        self.dataset_config = dataset_config
        self.smiles_list: list[str] = smiles_list

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        embeddings = self.embeddings[index]
        padding_mask = self.padding_mask[index]

        # auxillary_data =

        regression_targets = self.regression_targets[index]
        regression_masks = self.regression_masks[index]

        return embeddings, padding_mask, regression_targets, regression_masks

    def store_data_to_disk(self, directory):
        # ensure that directory exists:
        os.makedirs(directory, exist_ok=True)

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

        np.save(f"{directory}/padded_positions.npy", padded_positions)
        np.save(f"{directory}/padding_dim.npy", padding_dim)
        np.save(f"{directory}/padded_atomic_numbers.npy", padded_atomic_numbers)
        np.save(f"{directory}/regression_targets.npy", self.regression_targets)
        np.save(f"{directory}/regression_masks.npy", self.regression_masks)
        with open(f"{directory}/smiles_list", "w") as f:
            for i in self.smiles_list:
                f.write(i + "\n")

        to_yaml(f"{directory}/dataset_config.yaml", self.dataset_config)

    def normalize_targets(self, mean=None, std=None):
        # Normalize the regression targets

        regression_targets_cpu = self.regression_targets.detach().cpu().numpy()
        if mean is None:
            mean = np.mean(regression_targets_cpu, axis=0, where=self.regression_masks)
        if std is None:
            std = np.std(regression_targets_cpu, axis=0, where=self.regression_masks)

        self.regression_targets = (self.regression_targets - mean) / std
        self.dataset_config.mean = mean
        self.dataset_config.std = std

    def calculate_embeddings(self, calculator: MACECalculator, embedding_size: int):
        embeddings = np.zeros(
            shape=(
                self.dataset_config.N_molecules,
                self.dataset_config.max_atoms,
                embedding_size,
            )
        )
        padding_mask = np.ones(
            shape=(self.dataset_config.N_molecules, self.dataset_config.max_atoms)
        )  # Integer 1 = Boolean True = means that this position is padding

        for i, atoms in enumerate(self.molecules):
            descriptors = calculator.get_descriptors(atoms, invariants_only=False)
            num_atoms = len(atoms.get_atomic_numbers())
            embeddings[i, :num_atoms, :] = descriptors
            padding_mask[i, :num_atoms] = 0

        self.embeddings = embeddings
        self.padding_mask = padding_mask


class DatasetFactory:
    @classmethod
    def from_disk(cls, directory: str):
        # Load dataset_config first
        dataset_config: DatasetConfig = from_yaml(
            f"{directory}/dataset_config.yaml", DatasetConfig
        )

        print(dataset_config)

        positions = np.load(f"{directory}/padded_positions.npy")
        padding_dim = np.load(f"{directory}/padding_dim.npy")
        atomic_numbers = np.load(f"{directory}/padded_atomic_numbers.npy")
        molecules = []

        for i, padding in enumerate(padding_dim):
            atom = Atoms(
                numbers=atomic_numbers[i, :padding],
                positions=positions[i, :padding, :].squeeze(),
            )
            molecules.append(atom)

        with open(directory + "/smiles_list") as f:
            smiles_list = f.readlines()
            smiles_list = [i.strip() for i in smiles_list]

        if (
            dataset_config.dataset_type == "Pretraining"
            or dataset_config.dataset_type == "Evaluation"
        ):
            return AtomEmbeddingDataset(molecules, dataset_config, smiles_list)

        elif dataset_config.dataset_type == "Regression":
            regression_targets_arr = np.load(directory + "/regression_targets.npy")
            regression_masks_arr = np.load(directory + "/regression_masks.npy")

            regression_targets = from_numpy(regression_targets_arr).float()
            regression_masks = from_numpy(regression_masks_arr).float()

            return RegressionAtomEmbeddingDataset(
                molecules,
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
        mace_caluclator: MACECalculator,
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
        molecules = []

        smiles_counter = 0
        data_points_counter = 0

        with tqdm(total=dataset_config.N_molecules) as pbar:
            while data_points_counter < dataset_config.N_molecules:
                try:
                    smiles = next(iterator)
                except StopIteration:
                    print(
                        f"Reached StopIteration prematurely. Completed reading {data_points_counter} molecules."
                    )
                    break

                try:
                    N_conformers = min(
                        dataset_config.N_conformers,
                        dataset_config.N_molecules - data_points_counter,
                    )  # This ensures that the dataloading does not overshoot the targeted number of molecules
                    relaxed_molecules = get_relaxed_conformers(
                        smiles, mace_caluclator, dataset_config, N_conformers
                    )
                    if len(relaxed_molecules) == 0:
                        print(smiles_counter)

                except ValueError as ve:
                    tqdm.write(
                        f"Error with Generating Conformers for Smiles {smiles}: {ve}"
                    )

                N_confs = len(relaxed_molecules)
                smiles_list.extend([smiles] * N_confs)
                index_list.extend([smiles_counter] * N_confs)
                molecules.extend(relaxed_molecules)
                data_points_counter += N_confs
                pbar.update(N_confs)

                smiles_counter = smiles_counter + 1

        print(
            f"Read a total of {data_points_counter} from {smiles_counter} distinct SMILES"
        )
        # Check whether the dataset actually contains the desired number of molecules
        num_molecules = len(molecules)
        dataset_config.N_molecules = num_molecules

        if (
            dataset_config.dataset_type == "Pretraining"
            or dataset_config.dataset_type == "Evaluation"
        ):
            return AtomEmbeddingDataset(molecules, dataset_config, smiles_list)
        elif dataset_config.dataset_type == "Regression":
            # get the regression targets for the molecules
            regression_target = regression_target[index_list, :]
            regression_masks = regression_masks[index_list, :]

            return RegressionAtomEmbeddingDataset(
                molecules,
                regression_target,
                regression_masks,
                dataset_config,
                smiles_list,
            )

    @classmethod
    def from_smiles_with_hdf5_caching(
        cls,
        iterator: SmilesIterator,
        mace_caluclator: MACECalculator,
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
