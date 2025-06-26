from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data as data
from ase import Atoms

if TYPE_CHECKING:
    # only for mypy / IDE - never executed at runtime
    from threedscriptors.configuration.data_config import DatasetConfig

from threedscriptors.data_handling.data_utils import (
    compute_splits,
    get_max_molecule_size_from_atoms,
    get_max_molecule_size_from_smiles,
    get_max_num_of_heavy_atom_from_atoms,
)
from threedscriptors.data_handling.sample import Sample
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.utils.model_utils import (
    get_invariant_indices,
    split_invariants_equivariants,
)

type Molecules = list[Atoms]


class BaseDataset(data.Dataset):
    def __init__(
        self,
        dataset_config: "DatasetConfig",
        smiles_list=None,
        mol_ids=None,
        molecules: list[Atoms] | None = None,
        embeddings=None,
        padding_mask=None,
        regression_targets=None,
        regression_masks=None,
        auxillary_data=None,
        target_class_labels=None,
        active_decoy_labels=None,
        molecular_descriptors=None,
        atomic_positions =None
    ):
        super().__init__()

        self.dataset_config = dataset_config

        # Create all the fields for the child classes, which allows us to unify the store data to disk classes

        self.smiles_list = smiles_list
        self.mol_ids = mol_ids
        self.molecules: list[Atoms] | None = molecules
        self.embeddings = embeddings
        self.padding_mask = padding_mask
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks
        self.auxillary_data = auxillary_data
        self.target_class_labels = target_class_labels
        self.active_decoy_labels = active_decoy_labels
        self.molecular_descriptors = molecular_descriptors
        self.atomic_positions = atomic_positions

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, _):
        return Sample()

    def to_torch(self):
        if self.embeddings is not None:
            self.embeddings = torch.Tensor(self.embeddings)
        if self.padding_mask is not None:
            self.padding_mask = torch.Tensor(self.padding_mask)
        if self.regression_masks is not None:
            self.regression_masks = torch.Tensor(self.regression_masks)
        if self.regression_targets is not None:
            self.regression_targets = torch.Tensor(self.regression_targets)
        if self.atomic_positions is not None:
            self.atomic_positions = torch.Tensor(self.atomic_positions)

    def get_max_atoms(self):
        if self.dataset_config.max_atoms is None:

            if self.dataset_config.only_heavy_atoms:
                assert self.molecules is not None
                self.dataset_config.max_atoms = get_max_num_of_heavy_atom_from_atoms(self.molecules)

            elif self.molecules is not None:
                self.dataset_config.max_atoms = get_max_molecule_size_from_atoms(
                    self.molecules
                )
            else:
                assert self.smiles_list is not None
                smiles_iterator = ListSmilesIterator(self.smiles_list)
                self.dataset_config.max_atoms = get_max_molecule_size_from_smiles(
                    smiles_iterator
                )



        return self.dataset_config.max_atoms



    def get_padded_positions(self):

        positions = [at.get_positions() for at in self.molecules]
        atomic_numbers = [at.get_atomic_numbers() for at in self.molecules]
        padding_dim = np.array([len(an) for an in atomic_numbers]) # The dimension of the real atoms, required to reconstruct whcich element are padding and which ones are not.

        if self.dataset_config.only_heavy_atoms:


            heavy_indices = [np.nonzero(nums != 1)[0].tolist() for nums in atomic_numbers]

            atomic_numbers = [
                nums[hi]                 # pick only the heavy Zs
                for nums, hi in zip(atomic_numbers, heavy_indices, strict=False)
            ]

            positions = [
                pos[hi]                  # pick only the heavy-atom rows (x,y,z)
                for pos, hi in zip(positions, heavy_indices, strict=False)
            ]

            # 4. new “padding dim” = number of heavies per mol
            padding_dim = np.array([len(hi) for hi in heavy_indices])


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
            ], dtype = np.float32
        )

        return padding_dim, padded_positions, padded_atomic_numbers

    def get_atomic_embedding_normalization_constants(self,input_irreps):

        if isinstance(self.padding_mask, torch.Tensor):
            padding_mask = self.padding_mask.cpu().numpy()
        else:
            padding_mask = self.padding_mask

        print(f"Embeddings_shape {self.embeddings.shape}")

        invariant_indices, _ = get_invariant_indices(input_irreps)
        invariant_embeddings, equivariant_embeddings = split_invariants_equivariants(self.embeddings, invariant_indices)

        if isinstance(invariant_embeddings, torch.Tensor):
            invariant_embeddings = invariant_embeddings.cpu().numpy()

        masks = np.where(np.expand_dims(padding_mask, axis=-1) == 0.0, True, False)

        inv_mean_per_dim, inv_std_per_dim = self.calculate_invariant_normalization_constants(invariant_embeddings, masks)


        equivariant_scale_factor = self.calculate_equivariant_scale_factor(equivariant_embeddings, masks)

        return inv_mean_per_dim, inv_std_per_dim, equivariant_scale_factor


    @staticmethod
    def calculate_invariant_normalization_constants(invariant_embeddings, masks):

        mean_per_dim = np.mean(invariant_embeddings, axis=(0, 1), keepdims=True, where=masks)

        std_per_dim = np.std(invariant_embeddings, axis=(0, 1), keepdims=True, where=masks)

        return torch.from_numpy(mean_per_dim), torch.from_numpy(std_per_dim)

    @staticmethod
    def calculate_equivariant_scale_factor(equivariant_embeddings, masks):

        masks = masks.squeeze(-1)
        real_atom_equivariant_emebddings = equivariant_embeddings[masks]

        vecs = real_atom_equivariant_emebddings.reshape(-1,3)

        eps = 1e-12
        mean_sq = torch.mean(vecs ** 2)          # E[x²] over all x, y, z
        scale   = 1.0 / torch.sqrt(mean_sq + eps)

        return scale

    def split_dataset(self, splitting_ratios):

        splitting_indices = compute_splits(
            self.dataset_config.N_molecules,
            splitting_ratios,
            self.dataset_config.N_conformers,
        )

        returned_splits = []

        for slice_indices in splitting_indices:

            dataset_split = self[slice_indices.start : slice_indices.stop]

            # Update the dataset config with new number of molecules
            new_dataset_config = self.dataset_config.model_copy(
                update={"N_molecules": slice_indices.stop - slice_indices.start}
            )

            new_dataset = self.dataset_config.dataset_type.value(
                dataset_config=new_dataset_config, **asdict(dataset_split)
            )

            new_dataset.molecules = self.molecules[
                slice_indices.start : slice_indices.stop
            ]

            new_dataset.mol_ids = self.mol_ids[slice_indices.start : slice_indices.stop]

            new_dataset.smiles_list = self.smiles_list[
                slice_indices.start : slice_indices.stop
            ]
            returned_splits.append(new_dataset)

        return returned_splits


class AtomicEmbeddingMixin:
    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        padding_mask = self.padding_mask[idx]
        # get the rest of the tuple from next in MRO
        sample: Sample = super().__getitem__(idx)
        # prepend or append as you like
        sample.embeddings = emb
        sample.padding_mask = padding_mask
        return sample


class RegressionTargetMixin:
    def __getitem__(self, idx):
        sample: Sample = super().__getitem__(idx)

        regression_targets = self.regression_targets[idx]
        regression_masks = self.regression_masks[idx]

        sample.regression_targets = regression_targets
        sample.regression_masks = regression_masks

        return sample


class AtomicPositionMixin:
    def __getitem__(self, idx):
        pos = self.atomic_positions[idx]
        sample: Sample = super().__getitem__(idx)
        sample.atomic_positions = pos
        return sample


class AuxDataMixin:
    def __getitem__(self, idx):
        sample: Sample = super().__getitem__(idx)

        auxillary_data = {
            k: self.auxillary_data[k][idx, :] for k in self.auxillary_data.keys()
        }

        sample.auxillary_data = auxillary_data

        return sample


class SimilarityScreeningMixin:
    def __getitem__(self, idx):
        sample: Sample = super().__getitem__(idx)

        sample.active_decoy_labels = self.active_decoy_labels[idx]
        sample.target_class_labels = self.target_class_labels[idx]
        return sample


class MolecularDescriptorMixin:
    def __getitem__(self: BaseDataset, idx):
        sample: Sample = super().__getitem__(idx)

        sample.molecular_descriptors = self.molecular_descriptors[idx]

        return sample


class AtomicEmbeddingDataset(AtomicEmbeddingMixin, BaseDataset):
    pass


class AtomicEmbeddingWithPositionsDataset(AtomicPositionMixin,AtomicEmbeddingMixin, BaseDataset):
    pass


class RegressionDataset(RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset):
    pass


class RegressionDatasetwithPositions(AtomicPositionMixin,RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset):
    pass



class RegressionWithAuxAndPositionsDataset(
    AtomicPositionMixin,AuxDataMixin, RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset
):
    pass


class RegressionWithAuxDataset(
    AuxDataMixin, RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset
):
    pass


class SimilarityScreeningDataset(
    SimilarityScreeningMixin, MolecularDescriptorMixin, BaseDataset
):
    pass
