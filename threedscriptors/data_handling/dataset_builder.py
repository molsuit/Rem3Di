from dataclasses import dataclass
from math import ceil

import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator
from torch import from_numpy
from tqdm import tqdm

from threedscriptors.configuration.config_utils import from_yaml
from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.data_utils import (
    get_ase_atoms_with_conformers,
    get_mirrored_molecules,
    has_task_with_auxillary_data,
    relax_atoms,
)
from threedscriptors.data_handling.dataset import (
    AtomEmbeddingDataset,
    BaseAtomicDataset,
    RegressionAtomEmbeddingDataset,
    RegressionAtomEmbeddingDatasetWithAuxillaryData,
)
from threedscriptors.data_handling.smiles_iterator import SmilesIterator
from threedscriptors.utils.model_utils import get_mace_calculator_embedding_dimension


class DatasetBuilder:
    def __init__(self, dataset: BaseAtomicDataset, index_list: list[int]):
        self.dataset = dataset
        self.index_list = index_list

    @classmethod
    def load_initial_data_from_disk(cls, directory):
        # Load dataset_config first
        dataset_config: DatasetConfig = from_yaml(
            f"{directory}/dataset_config.yaml", DatasetConfig
        )

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

        with open(directory + "/smiles_list") as f:
            smiles_list = f.readlines()
            smiles_list = [i.strip() for i in smiles_list]

        dataset = BaseAtomicDataset(
            molecules=molecules, dataset_config=dataset_config, smiles_list=smiles_list
        )

        index_list = DatasetBuilder.get_index_list_from_smiles_list(smiles_list)
        # construct index_list aswell from smiles_list, better make index list a property with setter from smiles, getter

        return cls(dataset=dataset, index_list=index_list)

    @staticmethod
    def get_index_list_from_smiles_list(smiles_list: list[str]):
        string_to_id = {}
        result_ids = []
        current_id = 0

        # Process each string in the list
        for s in smiles_list:
            if s not in string_to_id:
                # Assign a new integer if the string has not been seen before
                string_to_id[s] = current_id
                current_id += 1
            # Append the mapped integer
            result_ids.append(string_to_id[s])

        return result_ids

    @classmethod
    def load_structures_from_smiles(
        cls,
        iterator: SmilesIterator,
        dataset_config: DatasetConfig,
    ):
        # Creates the conformal embeddings from smiles

        smiles_list = []
        index_list = []  # index list describes to which smiles index a datapoint belongs.
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

                    embeded_molecules = get_ase_atoms_with_conformers(
                        smiles, N_conformers
                    )
                    if len(embeded_molecules) == 0:
                        print(f"Error Embedding Smiles {smiles}, No. {smiles_counter}")

                except ValueError as ve:
                    tqdm.write(
                        f"Error with Generating Conformers for Smiles {smiles}: {ve}"
                    )

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
        return cls(
            dataset=BaseAtomicDataset(
                molecules=molecules,
                dataset_config=dataset_config,
                smiles_list=smiles_list,
            ),
            index_list=index_list,
        )

    @classmethod
    def load_pairwise_chiral_structures_from_smiles(
        cls, iterator: SmilesIterator, dataset_config: DatasetConfig
    ):
        smiles_list = []
        index_list = []  # index list describes to which smiles index a datapoint belongs.
        molecules = []

        smiles_counter = 0

        data_points_counter = 0

        mace_calculator = MACECalculator(
            dataset_config.embedding_model, device="cuda", enable_cueq=True
        )

        ### Returns a dataset with already relaxed (and mirrored!!!) Structures

        # Somewhere there should be an assert that odd features change sign...

        with tqdm(total=dataset_config.N_molecules) as pbar:
            while data_points_counter < dataset_config.N_molecules:
                try:
                    smiles_0 = next(iterator)
                    smiles_1 = next(iterator)
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
                        dataset_config.N_molecules - data_points_counter,
                    )  # This ensures that the dataloading does not overshoot the targeted number of molecules

                    N_conformers_per_enantiomer = int(ceil(total_N_conformers / 2))

                    embeded_molecules_0 = get_ase_atoms_with_conformers(
                        smiles_0, N_conformers_per_enantiomer
                    )
                    if len(embeded_molecules_0) == 0:
                        print(
                            f"Error Embedding Smiles {smiles_0}, No. {smiles_counter}"
                        )
                        raise ValueError()

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
        return cls(
            dataset=BaseAtomicDataset(
                molecules=molecules,
                dataset_config=dataset_config,
                smiles_list=smiles_list,
            ),
            index_list=index_list,
        )

    def relax_structures(self, mace_calculator: MACECalculator):
        # TODO: I recently saw https://github.com/Radical-AI/torch-sim, which could speed up the relaxation, because it batches ase atoms. Speedup they give is 18x for 108 atoms per molecule on a H100, much faster for smaller systems.(up to 100x for batch of 16 atoms systems )

        failed_relaxations = []

        for idx, molecule in enumerate(self.dataset.molecules):
            try:
                relax_atoms(
                    molecule,
                    mace_calculator,
                    self.dataset.dataset_config.BFGS_tol,
                    self.dataset.dataset_config.BFGS_max_steps,
                )
            except ValueError:
                print("Molecule did not relax.")
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
            self.index_list = [self.index_list[i] for i in correct_molecule_indices]

        self.dataset.dataset_config.has_relaxed_positions = True

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

        for i, atoms in enumerate(self.dataset.molecules):
            descriptors = calculator.get_descriptors(atoms, invariants_only=False)
            num_atoms = len(atoms.get_atomic_numbers())
            embeddings[i, :num_atoms, :] = descriptors
            padding_mask[i, :num_atoms] = 0

        self.dataset.embeddings = embeddings
        self.dataset.padding_mask = padding_mask

        self.dataset.dataset_config.has_atomic_embeddings = True

    def reload_atomic_embeddings(self, directory: str):
        self.dataset.embeddings = np.load(f"{directory}/embeddings.npy")
        self.dataset.padding_mask = np.load(f"{directory}/padding_mask.npy")

    def add_regression_data(
        self,
        regression_targets: np.ndarray | None = None,
        regression_masks: np.ndarray | None = None,
    ):
        assert self.index_list is not None

        # Transform the regression labels from 1 per smiles to 1 per conformer
        if regression_targets.ndim == 1:
            self.dataset.regression_targets = regression_targets[self.index_list]
            self.dataset.regression_masks = regression_masks[self.index_list]
        elif regression_targets.ndim == 2:
            self.dataset.regression_targets = regression_targets[self.index_list, :]
            self.dataset.regression_masks = regression_masks[self.index_list, :]
        else:
            raise ValueError(
                "Regression Target Array has unexpected numbers of dimensions"
            )

    def reload_regression_data(self, directory):
        regression_targets_arr = np.load(directory + "/regression_targets.npy")
        regression_masks_arr = np.load(directory + "/regression_masks.npy")

        self.dataset.regression_targets = from_numpy(regression_targets_arr).float()
        self.dataset.regression_masks = from_numpy(regression_masks_arr).float()

    def add_auxillary_data(self, auxillary_data: dict[str : np.ndarray]):
        expanded_aux_dict = {}
        # auxillary data needs to be expanded to have data for every conformer
        for task, aux_data in auxillary_data.items():
            expanded_aux_data = aux_data[self.index_list, :]
            expanded_aux_dict[task] = expanded_aux_data
            self.dataset.auxillary_data = expanded_aux_dict

    def reload_auxillary_data(self, directory):
        np_auxillary_data = np.load(f"{directory}/auxillary_data.npz")

        aux_keys = np_auxillary_data.files

        self.dataset.auxillary_data = {key: np_auxillary_data[key] for key in aux_keys}

    def get_dataset(self):
        # This differentiation between differen Datasets is necessary because each dataset class implements its own __getitem__ function, which is used in the dataloader.
        # Is there a more elegant way to get the child classs from the base atomic dataset ?

        if self.dataset.auxillary_data is not None:
            return RegressionAtomEmbeddingDatasetWithAuxillaryData(
                molecules=self.dataset.molecules,
                dataset_config=self.dataset.dataset_config,
                smiles_list=self.dataset.smiles_list,
                embeddings=self.dataset.embeddings,
                padding_mask=self.dataset.padding_mask,
                regression_targets=self.dataset.regression_targets,
                regression_masks=self.dataset.regression_masks,
                auxillary_data=self.dataset.auxillary_data,
            )

        elif self.dataset.regression_targets is not None:
            return RegressionAtomEmbeddingDataset(
                molecules=self.dataset.molecules,
                dataset_config=self.dataset.dataset_config,
                smiles_list=self.dataset.smiles_list,
                embeddings=self.dataset.embeddings,
                padding_mask=self.dataset.padding_mask,
                regression_targets=self.dataset.regression_targets,
                regression_masks=self.dataset.regression_masks,
            )
        elif self.dataset.embeddings is not None:
            return AtomEmbeddingDataset(
                molecules=self.dataset.molecules,
                dataset_config=self.dataset.dataset_config,
                smiles_list=self.dataset.smiles_list,
                embeddings=self.dataset.embeddings,
                padding_mask=self.dataset.padding_mask,
            )
        elif self.dataset.molecules is not None:
            return BaseAtomicDataset(
                molecules=self.dataset.molecules,
                dataset_config=self.dataset.dataset_config,
                smiles_list=self.dataset.smiles_list,
            )
        else:
            raise ValueError("No dataset to return from the dataset builder.")


@dataclass
class BuildConfiguration:
    build_atomic_embeddings: bool = False
    build_regression_targets: bool = False
    build_auxillary_data: bool = False
    relax_molecule_structure: bool = False


class DatasetBuildingDirector:
    def __init__(self, builder: DatasetBuilder):
        self.builder = builder
        self.config = builder.dataset.dataset_config

    @staticmethod
    def check_config(config: DatasetConfig):
        # Check the config for which fields to construct
        construction_recepie = BuildConfiguration()
        if config.tasks is None:
            construction_recepie.build_atomic_embeddings = True
        elif config.tasks is not None:
            construction_recepie.build_atomic_embeddings = True
            construction_recepie.build_regression_targets = True

            if has_task_with_auxillary_data(config.tasks):
                construction_recepie.build_auxillary_data = True

        return construction_recepie

    @classmethod
    def build_dataset(
        cls,
        iterator: SmilesIterator,
        dataset_config: DatasetConfig,
        regression_targets=None,
        regression_masks=None,
        auxillary_data=None,
        return_normalized_targets: bool = False,
        return_normalized_inputs: bool = False,
    ):
        # use the builder to assemble the dataset according to the configuration

        builder = DatasetBuilder.load_structures_from_smiles(
            iterator=iterator, dataset_config=dataset_config
        )

        construction_recepie = cls.check_config(builder.dataset.dataset_config)
        if construction_recepie.build_atomic_embeddings:
            assert dataset_config.embedding_model is not None

            embedding_model = MACECalculator(
                model_paths=dataset_config.embedding_model,
                device="cuda",
                enable_cueq=True,
            )
            # Is there ever a point where we do not want to relax the structures?
            builder.relax_structures(embedding_model)

            builder.calculate_atomic_embeddings(
                calculator=embedding_model,
                embedding_size=get_mace_calculator_embedding_dimension(embedding_model),
            )

        if construction_recepie.build_regression_targets:
            assert regression_masks is not None and regression_targets is not None

            builder.add_regression_data(
                regression_targets=regression_targets, regression_masks=regression_masks
            )

        if construction_recepie.build_auxillary_data:
            assert auxillary_data
            builder.add_auxillary_data(auxillary_data=auxillary_data)

        dataset = builder.get_dataset()

        if return_normalized_targets:
            assert isinstance(dataset, RegressionAtomEmbeddingDataset)

            dataset.normalize_regression_targets()

        if return_normalized_inputs:
            dataset.normalize_atomic_embeddings()

        dataset.to_torch()

        return cls(builder=builder), dataset

    @classmethod
    def reload_dataset(
        cls,
        directory,
        return_normalized_targets: bool = False,
        return_normalized_inputs: bool = False,
    ):
        builder = DatasetBuilder.load_initial_data_from_disk(directory)

        construction_recepie = cls.check_config(builder.dataset.dataset_config)

        if construction_recepie.build_atomic_embeddings:
            builder.reload_atomic_embeddings(directory=directory)
        if construction_recepie.build_regression_targets:
            builder.reload_regression_data(directory=directory)
        if construction_recepie.build_auxillary_data:
            builder.reload_auxillary_data(directory=directory)

        dataset = builder.get_dataset()

        if return_normalized_targets:
            assert isinstance(dataset, RegressionAtomEmbeddingDataset)
            dataset.normalize_regression_targets()

        if return_normalized_inputs:
            dataset.normalize_atomic_embeddings()

        dataset.to_torch()

        return cls(builder=builder), dataset

    @classmethod
    def build_chiral_dataset(
        cls,
        iterator: SmilesIterator,
        dataset_config: DatasetConfig,
        regression_targets=None,
        regression_masks=None,
        auxillary_data=None,
        return_normalized_targets: bool = False,
        return_normalized_inputs: bool = False,
    ):
        builder = DatasetBuilder.load_pairwise_chiral_structures_from_smiles(
            iterator=iterator, dataset_config=dataset_config
        )

        construction_recepie = cls.check_config(builder.dataset.dataset_config)

        if construction_recepie.build_atomic_embeddings:
            assert dataset_config.embedding_model is not None

            embedding_model = MACECalculator(
                model_paths=dataset_config.embedding_model,
                device="cuda",
                enable_cueq=True,
            )

            builder.calculate_atomic_embeddings(
                calculator=embedding_model,
                embedding_size=get_mace_calculator_embedding_dimension(embedding_model),
            )

        if construction_recepie.build_regression_targets:
            assert regression_masks is not None and regression_targets is not None

            builder.add_regression_data(
                regression_targets=regression_targets, regression_masks=regression_masks
            )

        if construction_recepie.build_auxillary_data:
            assert auxillary_data
            builder.add_auxillary_data(auxillary_data=auxillary_data)

        dataset = builder.get_dataset()

        if return_normalized_targets:
            assert isinstance(dataset, RegressionAtomEmbeddingDataset)

            dataset.normalize_regression_targets()

        if return_normalized_inputs:
            dataset.normalize_atomic_embeddings()

        dataset.to_torch()

        return cls(builder=builder), dataset

    @classmethod
    def custom_build(cls, recepie: BuildConfiguration):
        raise NotImplementedError
