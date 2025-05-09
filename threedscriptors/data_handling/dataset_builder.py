from math import ceil

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator
from tqdm import tqdm

from threedscriptors.data_handling.data_utils import (
    get_ase_atoms_with_conformers,
    get_mirrored_molecules,
    get_unique_smiles_id_from_smiles_list,
    relax_atoms,
)
from threedscriptors.data_handling.dataset import (
    BaseDataset,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator


class DatasetBuilder:
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def add_smiles_data(self, smiles):
        smiles_list = smiles

        # if self.dataset.dataset_config.N_molecules is not None:
        #    assert self.dataset.dataset_config.N_molecules == len(smiles_list)

        self.dataset.smiles_list = smiles_list
        self.dataset.mol_ids = get_unique_smiles_id_from_smiles_list(smiles_list)

    def add_molecules(self, molecules: list[Atoms], mol_ids: np.ndarray):
        assert mol_ids.shape[0] == len(molecules)
        self.dataset.molecules = molecules
        self.dataset.mol_ids = mol_ids

    def embed_structures_from_smiles(self):
        dataset_config = self.dataset.dataset_config

        initial_smiles_list = self.dataset.smiles_list

        if dataset_config.N_molecules is None:
            limit = len(initial_smiles_list)*dataset_config.N_conformers

        else:
            limit = dataset_config.N_molecules

        smiles_iterator = ListSmilesIterator(initial_smiles_list)

        # Creates the conformal embeddings from smiles
        smiles_list = []
        index_list = []  # index list describes to which smiles index a datapoint belongs.
        molecules = []

        smiles_counter = 0
        data_points_counter = 0

        with tqdm(total=limit) as pbar:
            while data_points_counter < limit:
                try:
                    smiles = next(smiles_iterator)
                except StopIteration:
                    print(
                        f"Reached StopIteration prematurely. Completed reading {data_points_counter} molecules."
                    )
                    break

                try:
                    N_conformers = min(
                        dataset_config.N_conformers,
                        limit - data_points_counter,
                    )  # This ensures that the dataloading does not overshoot the targeted number of molecules

                    embeded_molecules = get_ase_atoms_with_conformers(
                        smiles, N_conformers
                    )
                    if len(embeded_molecules) == 0:
                        print(f"Error Embedding Smiles {smiles}, No. {smiles_counter}")

                except ValueError as ve:
                    tqdm.write(
                        f"Error with Generating Conformers for Smiles {smiles}: {ve}"
                    )
                    continue

                N_confs = len(embeded_molecules)
                smiles_list.extend([smiles] * N_confs)
                index_list.extend([smiles_counter] * N_confs)
                molecules.extend(embeded_molecules)
                data_points_counter += N_confs
                pbar.update(N_confs)

                smiles_counter = smiles_counter + 1

        num_molecules = len(molecules)
        dataset_config.N_molecules = num_molecules

        print(
            f"Read a total of {data_points_counter} from {smiles_counter} distinct SMILES"
        )
        self.dataset.molecules = molecules
        self.dataset.smiles_list = smiles_list
        self.dataset.mol_ids = index_list

    def load_pairwise_chiral_structures_from_smiles(self):
        dataset_config = self.dataset.dataset_config

        initial_smiles_list = self.dataset.smiles_list

        if dataset_config.N_molecules is None:
            limit = len(initial_smiles_list) + 1

        else:
            limit = dataset_config.N_molecules

        smiles_iterator = ListSmilesIterator(initial_smiles_list)

        smiles_list = []
        index_list = []  # index list describes to which smiles index a datapoint belongs.
        molecules = []

        smiles_counter = 0

        data_points_counter = 0

        mace_calculator = dataset_config.embedding_model_config.mace_calc

        ### Returns a dataset with already relaxed (and mirrored!!!) Structures

        # Somewhere there should be an assert that odd features change sign...

        with tqdm(total=limit) as pbar:
            while data_points_counter < limit:
                try:
                    smiles_0 = next(smiles_iterator)
                    smiles_1 = next(smiles_iterator)
                    # pairwise iterator returns enantiomer pairs
                    print(smiles_0)
                    print(smiles_1)
                except StopIteration:
                    print(
                        f"Reached StopIteration prematurely. Completed reading {data_points_counter} molecules."
                    )
                    break

                try:
                    total_N_conformers = min(
                        dataset_config.N_conformers,
                        limit - data_points_counter,
                    )  # This ensures that the dataloading does not overshoot the targeted number of molecules

                    N_conformers_per_enantiomer = int(ceil(total_N_conformers / 2))

                    embeded_molecules_0 = get_ase_atoms_with_conformers(
                        smiles_0, N_conformers_per_enantiomer
                    )
                    if len(embeded_molecules_0) == 0:
                        raise ValueError(
                            f"Error Embedding Smiles {smiles_0}, No. {smiles_counter}"
                        )

                    # relax embedded_molecules
                    for mol in embeded_molecules_0:
                        relax_atoms(
                            mol,
                            mace_calculator,
                            BFGS_tol=dataset_config.BFGS_tol,
                            max_steps=dataset_config.BFGS_max_steps,
                        )

                    embeded_molecules_1 = get_mirrored_molecules(embeded_molecules_0)

                except ValueError as ve:
                    tqdm.write(
                        f"Error with Generating Conformers for Smiles {smiles_0}: {ve}"
                    )
                    continue

                print(smiles_0)
                print(smiles_1)
                N_confs_per_enantionmer = len(embeded_molecules_0)
                N_total_confs = 2 * N_confs_per_enantionmer

                smiles_list.extend(
                    [smiles_0] * N_confs_per_enantionmer
                    + [smiles_1] * N_confs_per_enantionmer
                )
                index_list.extend(
                    [smiles_counter] * N_confs_per_enantionmer
                    + [smiles_counter + 1] * N_conformers_per_enantiomer
                )
                molecules.extend(embeded_molecules_0 + embeded_molecules_1)
                data_points_counter += N_total_confs
                pbar.update(N_total_confs)

                smiles_counter = smiles_counter + 2

        num_molecules = len(molecules)
        dataset_config.N_molecules = num_molecules

        print(
            f"Read a total of {data_points_counter} from {smiles_counter} distinct SMILES"
        )
        self.dataset.molecules = molecules
        self.dataset.smiles_list = smiles_list
        self.dataset.mol_ids = index_list

    def relax_structures(self, mace_calculator: MACECalculator):
        assert self.dataset.molecules is not None
        failed_relaxations = []

        sucessfull_relaxations = []

        for idx, molecule in tqdm(
            enumerate(self.dataset.molecules), total=len(self.dataset.molecules)
        ):
            if (
                len(sucessfull_relaxations)
                >= self.dataset.dataset_config.N_molecules
                * self.dataset.dataset_config.N_conformers
            ):
                break

            try:
                relax_atoms(
                    molecule,
                    mace_calculator,
                    self.dataset.dataset_config.BFGS_tol,
                    self.dataset.dataset_config.BFGS_max_steps,
                )
                sucessfull_relaxations.append(idx)
            except ValueError:
                tqdm.write("Molecule did not relax.")
                # Remove the Molecule from self.molecule ?
                failed_relaxations.append(idx)
                continue

        # correct all data by removing molecules with failed relaxations

        if failed_relaxations != []:
            correct_molecule_indices = list(range(0, len(self.dataset.molecules)))

            correct_molecule_indices = [
                i for i in correct_molecule_indices if i not in failed_relaxations
            ]

            self.dataset.molecules = [
                self.dataset.molecules[i] for i in correct_molecule_indices
            ]

            self.dataset.smiles_list = [
                self.dataset.smiles_list[i] for i in correct_molecule_indices
            ]
            self.dataset.mol_ids = [
                self.dataset.mol_ids[i] for i in correct_molecule_indices
            ]

        else:
            self.dataset.mol_ids = [self.dataset.mol_ids[i] for i in sucessfull_relaxations]
            self.dataset.molecules = [
                self.dataset.molecules[i] for i in sucessfull_relaxations
            ]

        print(f"{len(self.dataset.molecules)} Molecules")

    def calculate_atomic_embeddings(
        self, calculator: MACECalculator, embedding_size: int
    ):
        # Assert that the mace caluclator of the embeddings and the relaxation are the same?
        # Check the maximum number of atoms in loaded smiles
        _ = self.dataset.get_max_atoms()

        embeddings = np.zeros(
            shape=(
                self.dataset.dataset_config.N_molecules,
                self.dataset.dataset_config.max_atoms,
                embedding_size,
            )
        )
        padding_mask = np.ones(
            shape=(
                self.dataset.dataset_config.N_molecules,
                self.dataset.dataset_config.max_atoms,
            )
        )  # Integer 1 = Boolean True = means that this position is padding

        for i, atoms in tqdm(
            enumerate(self.dataset.molecules), total=len(self.dataset.molecules)
        ):
            descriptors = calculator.get_descriptors(atoms, invariants_only=False)
            num_atoms = len(atoms.get_atomic_numbers())
            embeddings[i, :num_atoms, :] = descriptors
            padding_mask[i, :num_atoms] = 0

        self.dataset.embeddings = torch.Tensor(embeddings)
        self.dataset.padding_mask = torch.Tensor(padding_mask)

    def add_regression_data(
        self,
        regression_targets: np.ndarray | None = None,
        regression_masks: np.ndarray | None = None,
    ):
        assert self.dataset.mol_ids is not None

        # Transform the regression labels from 1 per smiles to 1 per conformer
        if regression_targets.ndim == 1:
            regression_targets = regression_targets[self.dataset.mol_ids]
            regression_masks = regression_masks[self.dataset.mol_ids]
        elif regression_targets.ndim == 2:
            regression_targets = regression_targets[self.dataset.mol_ids, :]
            regression_masks = regression_masks[self.dataset.mol_ids, :]
        else:
            raise ValueError(
                "Regression Target Array has unexpected numbers of dimensions"
            )

        assert np.all(np.any(regression_targets, axis=0)), "There are some empty tasks "
        self.dataset.regression_targets = torch.Tensor(regression_targets)
        self.dataset.regression_masks = torch.Tensor(regression_masks)

    def add_molecular_descriptor(self, descriptor_calculator):
        # This should be discussed, if this goes to the evaluation or already in the dataset.
        self.dataset.molecular_descriptors = descriptor_calculator(self.dataset)

    def add_similarity_screening_data(self, class_label_data, activity_data):
        assert self.dataset.mol_ids is not None
        self.dataset.activity_decoy_labels = activity_data[self.dataset.mol_ids]
        self.dataset.target_class_labels = class_label_data[self.dataset.mol_ids]

    def add_auxillary_data(self, auxillary_data: dict[str : np.ndarray]):
        expanded_aux_dict = {}
        # auxillary data needs to be expanded to have data for every conformer
        for task, aux_data in auxillary_data.items():
            expanded_aux_data = aux_data[self.dataset.mol_ids, :]
            expanded_aux_dict[task] = torch.Tensor(expanded_aux_data).float()

        self.dataset.auxillary_data = expanded_aux_dict

    def normalize_regression_targets(self):
        assert self.dataset.regression_targets is not None

        if self.dataset.dataset_config.regression_is_normalized:
            print("Dataset was already normalized")
            return

        regression_targets = self.dataset.regression_targets
        if isinstance(regression_targets, torch.Tensor):
            regression_targets = regression_targets.detach().cpu().numpy()

        mean = np.mean(regression_targets, axis=0, where=self.dataset.regression_masks)

        std = np.std(regression_targets, axis=0, where=self.dataset.regression_masks)

        self.dataset.regression_targets = (regression_targets - mean) / std

        for task, task_mean, task_std in zip(
            self.dataset.dataset_config.tasks, mean.tolist(), std.tolist(), strict=False
        ):

            task.mean = task_mean
            task.std = task_std


        self.dataset.dataset_config.regression_is_normalized = True

        self.dataset.regression_targets = torch.Tensor(self.dataset.regression_targets)

    def normalize_atomic_embeddings(self, mean_per_dim=None, std_per_dim=None):
        if isinstance(self.dataset.padding_mask, torch.Tensor):
            padding_mask = self.dataset.padding_mask.detach().cpu().numpy()
        else:
            padding_mask = self.dataset.padding_mask

        if isinstance(self.dataset.embeddings, torch.Tensor):
            embeddings = self.dataset.embeddings.detach().cpu().numpy()
        else:
            embeddings = self.dataset.embeddings

        masks = np.where(np.expand_dims(padding_mask, axis=-1) == 0.0, True, False)

        if mean_per_dim is None:
            mean_per_dim = np.mean(embeddings, axis=(0, 1), keepdims=True, where=masks)
        else:
            assert mean_per_dim.shape == embeddings.shape

        if std_per_dim is None:
            std_per_dim = np.std(embeddings, axis=(0, 1), keepdims=True, where=masks)
        else:
            assert std_per_dim.shape == embeddings.shape

        self.mean_atomic_embeddings = mean_per_dim
        self.std_atomic_embeddings = std_per_dim

        # embeddings = (embeddings - mean_per_dim) / std_per_dim

        self.dataset.embeddings = torch.Tensor(embeddings)
        self.dataset.padding_mask = torch.Tensor(padding_mask)

    def get_mean_and_std_embeddings(self):
        if hasattr(self, "mean_atomic_embeddings"):
            mean_embeddings = self.mean_atomic_embeddings
        else:
            mean_embeddings = None

        if hasattr(self, "std_atomic_embeddings"):
            std_embeddings = self.std_atomic_embeddings
        else:
            std_embeddings = None

        return mean_embeddings, std_embeddings
