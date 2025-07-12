from math import ceil
from threedscriptors.configuration.data_config import LabelScalingType

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
from rdkit import Chem

from threedscriptors.data_handling.dataset import (
    BaseDataset,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.utils.model_utils import get_mace_calculator_embedding_dimension
from rdkit.Chem import rdmolops


from threedscriptors.data_handling.mol_id import StructureID


class DatasetBuilder:
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def add_smiles_data(self, smiles):

        # 1. Canonicalize smiles
        assert self.dataset.smiles_list is None and self.dataset.structure_ids is None

        mol_ids = []
        smiles_list = []

        for smi_idx, smi in enumerate(smiles):

            canonical_smiles = Chem.CanonSmiles(smi, useChiral=True)

            smiles_list.append(canonical_smiles)
            mol_ids.append(
                StructureID(
                    structure_id=smi_idx,
                    canonical_smiles=canonical_smiles,
                    molecule_id=smi_idx,
                    smiles_id=smi_idx,
                    conformer_id=0,
                )
            )

        self.dataset.structure_ids = mol_ids
        self.dataset.smiles_list = smiles_list


    def add_molecules(self, molecules: list[Atoms], mol_ids: list[StructureID]):

        assert mol_ids.shape[0] == len(molecules)
        self.dataset.molecules = molecules
        self.dataset.mol_ids = mol_ids

    def embed_structures_from_smiles(self):
        dataset_config = self.dataset.dataset_config

        initial_smiles_list = self.dataset.smiles_list

        if dataset_config.N_molecules is None:
            N_mol_limit = len(initial_smiles_list) * dataset_config.N_conformers
        else:
            N_mol_limit = dataset_config.N_molecules

        smiles_iterator = ListSmilesIterator(initial_smiles_list)

        # Creates the conformal embeddings from smiles
        smiles_list = []
        index_list = (
            []
        )  # index list describes to which smiles index a datapoint belongs.
        molecules = []

        embedded_molecule_counter = 0
        running_structure_id = 0
        with tqdm(total=N_mol_limit) as pbar:
            for smiles_counter, smiles in enumerate(smiles_iterator):
                if embedded_molecule_counter >= N_mol_limit:
                    break

                try:
                    embeded_molecules = get_ase_atoms_with_conformers(
                        smiles, N_conformers=dataset_config.N_conformers
                    )

                except ValueError as ve:
                    tqdm.write(
                        f"Error with Generating Conformers for Smiles {smiles}: {ve}"
                    )
                    continue

                for conformer_id, mol in enumerate(embeded_molecules):
                    molecules.append(mol)
                    index_list.append(
                        StructureID(
                            structure_id=running_structure_id,
                            molecule_id=embedded_molecule_counter,
                            conformer_id=conformer_id,
                            smiles_id=smiles_counter,
                            canonical_smiles=smiles,
                        )
                    )
                    smiles_list.append(smiles)
                    running_structure_id += 1

                pbar.update(1)
                embedded_molecule_counter += 1

        dataset_config.N_molecules = len(molecules)
        print(
            f"Read a total of {dataset_config.N_molecules} from {smiles_counter} distinct SMILES"
        )
        self.dataset.molecules = molecules
        self.dataset.smiles_list = smiles_list
        self.dataset.structure_ids = index_list
        self.dataset.N_structures = len(index_list)

    def canonicalize_structure_ids(self):

        new_structure_ids = []

        for new_unique_s_id, old_id in enumerate(self.dataset.structure_ids):

            new_structure_ids.append(
                StructureID(
                    structure_id=new_unique_s_id,
                    smiles_id= old_id.smiles_id,
                    canonical_smiles=old_id.canonical_smiles,
                    molecule_id=old_id.molecule_id,
                    enantiomer_id=old_id.enantiomer_id,
                    conformer_id=old_id.conformer_id
                )
            )

        self.dataset.structure_ids = new_structure_ids

    def load_pairwise_chiral_structures_from_smiles(self):
        dataset_config = self.dataset.dataset_config

        initial_smiles_list = self.dataset.smiles_list

        if dataset_config.N_molecules is None:
            limit = len(initial_smiles_list) + 1

        else:
            limit = dataset_config.N_molecules

        smiles_iterator = ListSmilesIterator(initial_smiles_list)

        smiles_list = []
        index_list = (
            []
        )  # index list describes to which smiles index a datapoint belongs.
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

                    N_conformers_per_enantiomer = ceil(total_N_conformers / 2)

                    embeded_molecules_0, _ = get_ase_atoms_with_conformers(
                        smiles_0,
                        N_conformers_per_enantiomer,
                        load_adjacency_matrix=dataset_config.load_adjacency_matrix,
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

                N_confs_per_enantionmer = len(embeded_molecules_0)
                N_total_confs = 2 * N_confs_per_enantionmer

                smiles_0 = Chem.CanonSmiles(smiles_0)
                smiles_1 = Chem.CanonSmiles(smiles_1)
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

        print(f"{len(failed_relaxations)} relaxations have failed.")

        self.dataset.N_structures = len(sucessfull_relaxations)

        self.dataset.molecules = [
            self.dataset.molecules[i] for i in sucessfull_relaxations
        ]

        self.dataset.structure_ids = [
            self.dataset.structure_ids[i] for i in sucessfull_relaxations
        ]

    def calculate_atomic_embeddings(self, calculator: MACECalculator):

        embedding_size = get_mace_calculator_embedding_dimension(calculator)

        only_heavy_atoms = self.dataset.dataset_config.only_heavy_atoms
        # Assert that the mace caluclator of the embeddings and the relaxation are the same?
        # Check the maximum number of atoms in loaded smiles
        self.dataset.get_max_atoms()

        embeddings = np.zeros(
            shape=(
                self.dataset.N_structures,
                self.dataset.max_atoms,
                embedding_size,
            )
        )
        padding_mask = np.ones(
            shape=(
                self.dataset.N_structures,
                self.dataset.max_atoms,
            )
        )  # Integer 1 = Boolean True = means that this position is padding

        for i, atoms in tqdm(
            enumerate(self.dataset.molecules), total=len(self.dataset.molecules)
        ):
            descriptors = calculator.get_descriptors(atoms, invariants_only=False)

            atomic_numbers = atoms.get_atomic_numbers()

            if only_heavy_atoms:
                # Slices out only the atoms with atomic number != 1
                heavy_atoms_indices = np.argwhere(atomic_numbers > 1)

                descriptors = descriptors[heavy_atoms_indices, :].squeeze()

                num_atoms = len(heavy_atoms_indices)

            else:
                num_atoms = len(atomic_numbers)

            embeddings[i, :num_atoms, :] = descriptors
            padding_mask[i, :num_atoms] = 0

        self.dataset.embeddings = torch.Tensor(embeddings)
        self.dataset.padding_mask = torch.Tensor(padding_mask).bool()

    def add_regression_data(
        self,
        regression_targets: torch.Tensor | None = None,
        regression_masks: torch.Tensor | None = None,
    ):

        assert self.dataset.structure_ids is not None

        dataset_idx = [id.smiles_id for id in self.dataset.structure_ids]

        # Transform the regression labels from 1 per smiles to 1 per conformer
        if regression_targets.ndim == 1:
            regression_targets = regression_targets[dataset_idx]
            regression_masks = regression_masks[dataset_idx]
        elif regression_targets.ndim == 2:
            regression_targets = regression_targets[dataset_idx, :]
            regression_masks = regression_masks[dataset_idx, :]
        else:
            raise ValueError(
                "Regression Target Array has unexpected numbers of dimensions"
            )

        regression_targets = torch.Tensor(regression_targets)
        regression_masks = torch.Tensor(regression_masks)

        assert torch.all(
            torch.any(regression_targets, dim=0)
        ), "There are some empty tasks "

        self.dataset.regression_targets = regression_targets
        self.dataset.regression_masks = regression_masks

    def add_molecular_descriptor(self, descriptor_calculator):
        # This should be discussed, if this goes to the evaluation or already in the dataset.
        self.dataset.molecular_descriptors = descriptor_calculator(self.dataset)

    def add_similarity_screening_data(self, class_label_data, activity_data):
        assert self.dataset.mol_ids is not None
        self.dataset.active_decoy_labels = activity_data[self.dataset.mol_ids]
        self.dataset.target_class_labels = class_label_data[self.dataset.mol_ids]

    def add_auxillary_data(self, auxillary_data: dict[str : np.ndarray]):
        expanded_aux_dict = {}
        dataset_idx = [id.smiles_id for id in self.dataset.structure_ids]
        # auxillary data needs to be expanded to have data for every conformer
        for task, aux_data in auxillary_data.items():
            expanded_aux_data = aux_data[dataset_idx, :]
            expanded_aux_dict[task] = torch.Tensor(expanded_aux_data).float()

        self.dataset.auxillary_data = expanded_aux_dict

    def add_atomic_positions(self):

        assert self.dataset.molecules is not None
        _, padded_pos, _ = self.dataset.get_padded_positions()

        if isinstance(padded_pos, np.ndarray):
            padded_pos = torch.from_numpy(padded_pos)

        self.dataset.atomic_positions = padded_pos

    def add_random_walk_matrices(self):

        transition_mats = []

        max_atoms = self.dataset.get_max_atoms()
        assert not self.dataset.dataset_config.only_heavy_atoms

        for molecule in self.dataset.molecules:
            assert "smiles" in molecule.info.keys()

            smiles = molecule.info["smiles"]

            mol = Chem.MolFromSmiles(smiles)
            A = rdmolops.GetAdjacencyMatrix(mol)
            # Get the adjacency matrix,

            A_self = A + np.eye(A.shape[0])

            deg = A_self.sum(axis=1)
            D_inv = np.diag(1.0 / deg)
            T = D_inv @ A_self

            ## Calculate the degree matrix
            # D_inv = 1 / A.sum(axis = 1)
            ## Invert and multiply
            # T = (D_inv[:, None] * A)
            #
            N_atoms = T.shape[0]
            # Pad
            T_padded = torch.zeros(size=(max_atoms, max_atoms))
            T_padded[:N_atoms, :N_atoms] = torch.from_numpy(T)

            # Collect

            transition_mats.append(T_padded)

        self.dataset.random_walk_transition_matrix = torch.stack(transition_mats, dim=0)

        print(self.dataset.random_walk_transition_matrix.shape)
