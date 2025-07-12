import os
from pathlib import Path
from threedscriptors.data_handling.mol_id import StructureID
import numpy as np
from ase import Atoms
from torch import from_numpy

from threedscriptors.configuration.config_utils import from_yaml, to_yaml
from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.data_utils import (
    get_unique_smiles_id_from_smiles_list,
)
from threedscriptors.data_handling.dataset import BaseDataset


def store_data_to_disk(dataset: BaseDataset, directory: str):
    os.makedirs(directory, exist_ok=True)

    # Store Positions
    padding_dim, padded_positions, padded_atomic_numbers = (
        dataset.get_padded_positions()
    )

    np.save(f"{directory}/padded_positions.npy", padded_positions)
    np.save(f"{directory}/padding_dim.npy", padding_dim)
    np.save(f"{directory}/padded_atomic_numbers.npy", padded_atomic_numbers)

    # Store Embeddings
    if dataset.embeddings is not None:
        np.save(f"{directory}/embeddings.npy", dataset.embeddings)
        np.save(f"{directory}/padding_mask.npy", dataset.padding_mask)

    # Store Regression Targets
    if dataset.regression_targets is not None:
        np.save(f"{directory}/regression_targets.npy", dataset.regression_targets)
        np.save(f"{directory}/regression_masks.npy", dataset.regression_masks)

    # Store Auxillary Data
    if dataset.auxillary_data is not None:
        np.savez(f"{directory}/auxillary_data.npz", **dataset.auxillary_data)

    if dataset.active_decoy_labels is not None:
        np.save(f"{directory}/activity_labels.npy", dataset.active_decoy_labels)

    if dataset.target_class_labels is not None:
        np.save(f"{directory}/target_class_labels.npy", dataset.target_class_labels)

    # Store Smiles
    if dataset.smiles_list and dataset.structure_ids is not None:
        with open(f"{directory}/smiles", "w") as f:
            for smi, structure_id in zip(dataset.smiles_list, dataset.structure_ids):

                f.write(f"{structure_id.to_id_string()} {smi}" + "\n")


    if dataset.random_walk_transition_matrix is not None:
        rw_matrix = dataset.random_walk_transition_matrix.cpu().numpy()
        np.save(f"{directory}/random_walk_transition_matrix.npy", rw_matrix)

    # Store Config
    to_yaml(f"{directory}/dataset_config.yaml", dataset.dataset_config)


def load_data_from_disk(
    directory: str | Path,
    load_molecules: bool = True,
) -> BaseDataset:
    directory = str(directory)
    # Load dataset_config first
    dataset_config = from_yaml(f"{directory}/dataset_config.yaml", DatasetConfig)

    dataset_cls = dataset_config.dataset_type.value
    dataset = dataset_cls(dataset_config=dataset_config)

    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]

    if "padded_positions.npy" in files and load_molecules:
        # This could be used to directly load the data from e.g. GEOM Drugwithout relaxation or simply reload the data from the disk.

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
        dataset.molecules = molecules

    if "smiles" in files:
        smiles_list = []
        structure_ids = []

        with open(directory + "/smiles") as f:

            for line in f:
                line = line.strip()
                if not line:
                    continue
                id_str, smi = line.split(" ", 1)

                id = StructureID.from_id_string(id_str, smiles=smi)
                
                smiles_list.append(smi)
                structure_ids.append(id)

        dataset.smiles_list = smiles_list
        dataset.structure_ids = structure_ids

    if "regression_targets.npy" in files:
        assert "regression_masks.npy" in files
        regression_targets_arr = np.load(directory + "/regression_targets.npy")
        regression_masks_arr = np.load(directory + "/regression_masks.npy")

        dataset.regression_targets = from_numpy(regression_targets_arr).float()
        dataset.regression_masks = from_numpy(regression_masks_arr).float()

    if "embeddings.npy" in files:
        dataset.embeddings = from_numpy(np.load(f"{directory}/embeddings.npy"))
        dataset.padding_mask = from_numpy(
            np.load(f"{directory}/padding_mask.npy")
        ).bool()

    if "activity_labels.npy" in files:
        assert "target_class_labels.npy" in files
        dataset.active_decoy_labels = np.load(f"{directory}/activity_labels.npy")
        dataset.target_class_labels = np.load(f"{directory}/target_class_labels.npy")

    if "auxillary_data.npz" in files:
        np_auxillary_data = np.load(f"{directory}/auxillary_data.npz")

        aux_keys = np_auxillary_data.files

        dataset.auxillary_data = {
            key: from_numpy(np_auxillary_data[key]) for key in aux_keys
        }

    if "random_walk_transition_matrix.npy" in files:
        rws = np.load(f"{directory}/random_walk_transition_matrix.npy")
        dataset.random_walk_transition_matrix = from_numpy(rws).float()

    return dataset
