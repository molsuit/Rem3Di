from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from ase import Atoms

if TYPE_CHECKING:
    # only for mypy / IDE - never executed at runtime
    from threedscriptors.configuration.data_config import DatasetConfig

from threedscriptors.data_handling.data_utils import (
    count_atoms_from_ase,
    count_atoms_from_smiles,
)
from threedscriptors.data_handling.mol_id import StructureID
from threedscriptors.data_handling.sample import Sample
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator

type Molecules = list[Atoms]

class BaseDataset(data.Dataset):
    def __init__(
        self,
        dataset_config: "DatasetConfig",
        smiles_list=None,
        structure_ids: list[StructureID] | None =None,
        molecules: Molecules | None = None,
        embeddings=None,
        padding_mask=None,
        regression_targets=None,
        regression_masks=None,
        auxillary_data=None,
        target_class_labels=None,
        active_decoy_labels=None,
        molecular_descriptors=None,
        atomic_positions =None,
        random_walk_transition_matrix = None
    ):
        super().__init__()

        self.dataset_config = dataset_config

        # Create all the fields for the child classes, which allows us to unify the store data to disk classes

        self.smiles_list = smiles_list
        self.structure_ids = structure_ids
        self.molecules = molecules
        self.embeddings = embeddings
        self.padding_mask = padding_mask
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks
        self.auxillary_data = auxillary_data
        self.target_class_labels = target_class_labels
        self.active_decoy_labels = active_decoy_labels
        self.molecular_descriptors = molecular_descriptors
        self.atomic_positions = atomic_positions
        self.random_walk_transition_matrix = random_walk_transition_matrix


        self.max_atoms = None
        self.total_num_atoms = None
        self.N_structures = None

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, _):
        return Sample()

    def to_torch(self):
        if self.embeddings is not None:
            self.embeddings = torch.from_numpy(self.embeddings)
        if self.padding_mask is not None:
            self.padding_mask = torch.from_numpy(self.padding_mask)
        if self.regression_masks is not None:
            self.regression_masks = torch.from_numpy(self.regression_masks)
        if self.regression_targets is not None:
            self.regression_targets = torch.from_numpy(self.regression_targets)
        if self.atomic_positions is not None:
            self.atomic_positions = torch.from_numpy(self.atomic_positions)

    def get_max_atoms(self):
        if self.max_atoms is None:
            if self.dataset_config.max_atoms is not None:
                self.max_atoms = self.dataset_config.max_atoms

            elif self.molecules is not None:
                self.max_atoms, _ = count_atoms_from_ase(self.molecules, heavy_atoms_only= self.dataset_config.only_heavy_atoms)

            else:
                assert self.smiles_list is not None
                smiles_iterator = ListSmilesIterator(self.smiles_list)
                self.max_atoms, _ = count_atoms_from_smiles(
                    smiles_iterator, heavy_atoms_only= self.dataset_config.only_heavy_atoms
                )

            self.dataset_config.max_atoms = self.max_atoms

        return self.max_atoms


    def get_structure_ids_for_mol(self, mol_ids):
        mol_ids_set = set(mol_ids)
        return [idx for idx, id in enumerate(self.structure_ids) if id.molecule_id in mol_ids_set]

    def get_all_mol_ids(self):
    # preserves first-seen order (unlike set(...))
        seen, out = set(), []
        for sid in self.structure_ids:
            m = sid.molecule_id
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def get_total_number_of_atoms(self):

        if self.total_num_atoms is None:

            if self.molecules is not None:
                _, self.total_num_atoms = count_atoms_from_ase(self.molecules, heavy_atoms_only= self.dataset_config.only_heavy_atoms)

            else:
                assert self.smiles_list is not None

                smiles_iterator = ListSmilesIterator(self.smiles_list)
                _, self.total_num_atoms = count_atoms_from_smiles(
                        smiles_iterator, heavy_atoms_only= self.dataset_config.only_heavy_atoms
                    )

        return self.total_num_atoms

    def get_padded_positions(self):

        positions = [at.get_positions() for at in self.molecules]
        atomic_numbers = [at.get_atomic_numbers() for at in self.molecules]
        padding_dim = np.array([len(an) for an in atomic_numbers]) # The dimension of the real atoms, required to reconstruct whcich element are padding and which ones are not.


        max_atoms= self.get_max_atoms()


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
                    an, (0, max_atoms - len(an)), mode="constant"
                )
                for an in atomic_numbers
            ]
        )

        # For positions, assuming each position array has shape (n_atoms, 3)
        padded_positions = np.array(
            [
                np.pad(
                    pos,
                    ((0, max_atoms - pos.shape[0]), (0, 0)),
                    mode="constant",
                )
                for pos in positions
            ], dtype = np.float32
        )


        return padding_dim, padded_positions, padded_atomic_numbers



    def expand_embedding_num_atoms(self, new_max_num_atoms: int):
        # Method can be used to increase the "Sequence length" i.e the number of atoms in a molecule. So that the embeddings do not have to be recalculated.
        # Expand the padding mask and the atomic embeddings to the max dimension.

        padding_width = new_max_num_atoms - self.max_atoms

        assert padding_width > 0

        self.embeddings = F.pad(
            self.embeddings, pad=(0, 0, 0, padding_width), value=0
        )

        self.padding_mask = F.pad(
            self.padding_mask, pad=(0, padding_width), value=1
        )

        if self.atomic_positions is not None:

            #self.atomic_positions is (B,N,3)
            self.atomic_positions = F.pad(self.atomic_positions, pad = (0, 0,0, padding_width), value = 0.0)


        self.max_atoms = new_max_num_atoms

    def convert_to_dataset_type(self, dataset_cls):

        def infer_required_fields(dataset_cls):
            fields = set( ['structure_ids']) # Structure ids is required
            for base in dataset_cls.__mro__: # Gets the inheritance order
                if hasattr(base, "required_fields"):

                    fields.update(base.required_fields)
            return fields


        req = infer_required_fields(dataset_cls)

        filtered = {}
        for k in req:
            v = getattr(self,k)
            filtered.update({k:v})

        return dataset_cls(dataset_config=self.dataset_config, **filtered)

class EnantiomerPairMixin:
    """
    Makes the dataset index over *pairs* of enantiomers.
    Each __getitem__(i) returns (sample_e1, sample_e2).

    Assumptions:
      - self.structure_ids is aligned with dataset storage (same order as tensors).
      - Enantiomers share (molecule_id, conformer_id) and differ by enantiomer_id in {1,2}.
    """

    required_fields = ["structure_ids"]

    def __init__(self, *args, **kwargs):
            """
            pair_by_conformer:
                If True, pairs are formed per (molecule_id, conformer_id).
                If False, pairs are formed per molecule_id (use only if you know each molecule has exactly one R/S).

            include_singletons:
                If True, unpaired entries (missing partner) are kept and mapped to None.
                If False, they are dropped from the index (recommended).
            """
            super().__init__(*args, **kwargs)
            self._build_enantiomer_index()

    def get_enantiomer_idx(self, idx: int) -> int | None:
        """Return the partner dataset index for a given dataset index, or None if none exists."""
        return self._partner_of.get(idx, None)

    def get_pair_indices(self, pair_idx: int) -> tuple[int, int | None]:
        """Return (idx_e1, idx_e2) for a pair index."""
        return self._pair_index[pair_idx]

    # ---------- Dataset overrides ----------
    def __len__(self) -> int:
        # number of *pairs* exposed by this mixin
        return len(self._pair_index)

    # ---------- internal ----------
    def _build_enantiomer_index(self):

        # Group dataset indices by key: (molecule_id, conformer_id) or molecule_id
        groups: dict[tuple[int, int | None], dict[int, int]] = {}

        for idx, sid in enumerate(self.structure_ids):
            assert idx == sid.structure_id
            # Skip entries with no enantiomer label
            if sid.enantiomer_id not in (1, 2):
                continue
            key = (sid.molecule_id, sid.conformer_id)
            d = groups.setdefault(key, {})
            d[sid.enantiomer_id] = idx


        self._pair_index: list[tuple[int, int]] = []
        self._partner_of: dict[int, int] = {}

        # Build symmetric mapping + the compact list of unique pairs
        for _, d in groups.items():
            i1 = d.get(1, None)
            i2 = d.get(2, None)

            if i1 is not None and i2 is not None:
                # Record pair (always [1] first for determinism)
                self._pair_index.append((i1, i2))
                self._partner_of[i1] = i2
                self._partner_of[i2] = i1
            else:
                raise ValueError


    def __getitem__(self,pair_idx):
        idx_e1, idx_e2 = self.get_pair_indices(pair_idx)
        # Always return first as enantiomer_id==1 if available (stable ordering)
        sample_e1 = super().__getitem__(idx_e1)
        sample_e2 = super().__getitem__(idx_e2)

        return sample_e1, sample_e2


    def get_pair_indices_for_structures(self, structure_id_set: set[int]) -> list[int]:
        """
        Return indices of *pairs* whose two structures are both inside `structure_id_set`.
        """
        return [
            pair_idx
            for pair_idx, (s1, s2) in enumerate(self._pair_index)
            if (s1 in structure_id_set) and (s2 in structure_id_set)
        ]

class AtomicEmbeddingMixin:
    required_fields = ["embeddings","padding_mask"]
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
    required_fields = ["regression_targets","regression_masks"]

    def __getitem__(self, idx):
        sample: Sample = super().__getitem__(idx)

        regression_targets = self.regression_targets[idx]
        regression_masks = self.regression_masks[idx]

        sample.regression_targets = regression_targets
        sample.regression_masks = regression_masks

        return sample


class AtomicPositionMixin:
    required_fields = ["atomic_positions"]

    def __getitem__(self, idx):
        pos = self.atomic_positions[idx]
        sample: Sample = super().__getitem__(idx)
        sample.atomic_positions = pos
        return sample


class AuxDataMixin:
    required_fields = ["auxillary_data"]

    def __getitem__(self, idx):
        sample: Sample = super().__getitem__(idx)

        auxillary_data = {
            k: self.auxillary_data[k][idx, :] for k in self.auxillary_data.keys()
        }

        sample.auxillary_data = auxillary_data

        return sample


class SimilarityScreeningMixin:
    required_fields = ["active_decoy_labels","target_class_labels"]

    def __getitem__(self, idx):
        sample: Sample = super().__getitem__(idx)

        sample.active_decoy_labels = self.active_decoy_labels[idx]
        sample.target_class_labels = self.target_class_labels[idx]

        return sample


class MolecularDescriptorMixin:
    required_fields = ["molecular_descriptors"]


    def __getitem__(self: BaseDataset, idx):
        sample: Sample = super().__getitem__(idx)

        sample.molecular_descriptors = self.molecular_descriptors[idx]

        return sample


class RandomWalkMixin:
    required_fields = ["random_walk_transition_matrix"]

    def __getitem__(self: BaseDataset, idx):

        sample: Sample = super().__getitem__(idx)

        sample.random_walk_transition_matrix = self.random_walk_transition_matrix[idx]

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


class PairedRegressionWithAuxAndPositionDataset(
    EnantiomerPairMixin,AtomicPositionMixin,AuxDataMixin, RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset
):
    pass

class RegressionWithAuxDataset(
    AuxDataMixin, RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset
):
    pass


class SimilarityScreeningDataset(
    SimilarityScreeningMixin, AtomicPositionMixin, AtomicEmbeddingMixin, BaseDataset
):
    pass


class RegressionDatasetwithRandomWalks(RandomWalkMixin,RegressionTargetMixin, AtomicEmbeddingMixin, BaseDataset):
    pass
