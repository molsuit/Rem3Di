import numpy as np
import torch
import torch.utils.data as data
from ase import Atoms

from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.data_utils import (
    get_max_molecule_size_from_atoms,
    get_max_molecule_size_from_smiles,
)
from threedscriptors.data_handling.sample import Sample
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator

type Molecules = list[Atoms]


class BaseDataset(data.Dataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
    ):
        super().__init__()

        self.dataset_config = dataset_config

        # Create all the fields for the child classes, which allows us to unify the store data to disk classes

        self.smiles_list = None
        self.mol_ids = None
        self.molecules: list[Atoms] | None = None
        self.embeddings = None
        self.padding_mask = None
        self.regression_targets = None
        self.regression_masks = None
        self.auxillary_data = None
        self.target_class_labels = None
        self.activity_decoy_labels = None
        self.molecular_descriptors = None

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

    def get_max_atoms(self):
        if self.dataset_config.max_atoms is None:
            if self.molecules is not None:
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

        sample.molecular_descriptor = self.molecular_descriptor[idx]

        return sample


class AtomicEmbeddingDataset(AtomicEmbeddingMixin, BaseDataset):
    pass


class RegressionDataset(RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset):
    pass


class RegressionWithAuxDataset(
    AuxDataMixin, RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset
):
    pass


class SimilarityScreeningDataset(
    SimilarityScreeningMixin, MolecularDescriptorMixin, BaseDataset
):
    pass
