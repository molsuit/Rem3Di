from collections.abc import Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from threedscriptors.configuration.training_config import SplitConfig, SplitStrategy
from threedscriptors.data_handling.data_utils import compute_splits
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset


class DatasetSplitting:
    def __init__(self, dataset: MoleculeDataset):
        self.dataset = dataset

    def _single_split(self, train_val_split_ratios, shuffle):

        # 1. Get the set of mol ids
        mol_ids = list(set(self.dataset.molecule_ids[:]))
        N_mols = len(mol_ids)

        # 2 Get random permutation if shuffle true

        if shuffle:
            # perm =  list that contains randomly shuffeld indices
            rng = np.random.default_rng(seed= 1)
            perm = rng.permutation(N_mols).tolist()
        else:
            perm = list(range(N_mols))

        # Split the mol ids

        splitting_indices = compute_splits(N_mols, train_val_split_ratios)

        train_perm = perm[splitting_indices[0].start : splitting_indices[0].stop]
        train_mol_ids = [mol_ids[i] for i in train_perm]

        val_perm = perm[splitting_indices[1].start : splitting_indices[1].stop]
        val_mol_ids = [mol_ids[i] for i in val_perm]

        # 4 get the correct struture_ids


        train_structure_ids = self.dataset.get_structure_ids_from_molecule_ids(train_mol_ids)

        validation_structure_ids = self.dataset.get_structure_ids_from_molecule_ids(val_mol_ids)


        #if self.dataset.regression_targets is not None:
        #    print(f"lables per class Train set{self.dataset.regression_masks[train_structure_ids,:].sum(dim= 0)}")
        #    print(f"lables per class Vals set{self.dataset.regression_masks[validation_structure_ids,:].sum(dim= 0)}")


        return train_structure_ids, validation_structure_ids, "Train"

    def _repeated_cv(
        self,
        N_repeats: int,
        N_splits: int,
        shuffle: bool = True,
    ):
        """
        Yields repeated k-fold (train_ids, val_ids) splits.

        Parameters
        ----------
        N_repeats : int
            Number of independent shuffles of the whole dataset.
        N_splits : int
            Number of folds (k in k-fold CV).
        shuffle : bool
            Shuffle molecules before each repeat.

        Yields
        ------
        Tuple[List[int], List[int]]
            train_structure_ids,  validation_structure_ids
        """
        # --- 1. Static info --------------------------------------------------
        mol_ids = self.dataset.get_all_mol_ids()
        n_mols = len(mol_ids)

        # Pre‑compute slice objects that divide an array into `N_splits` chunks
        fold_slices = compute_splits(n_mols, [1 / N_splits] * N_splits)

        # --- 2. Repeated CV ---------------------------------------------------
        for i_repeat in range(N_repeats):

            # 2a. (Re)shuffle indices
            if shuffle:
                rng = np.random.default_rng(seed = 1)
                perm = rng.permutation(n_mols).tolist()
            else:
                perm = list(range(n_mols))

            # 2b. Materialise indices for every fold
            folds = [perm[s.start : s.stop] for s in fold_slices]

            # 2c. Iterate over folds: each becomes validation once
            for val_fold_idx in range(N_splits):
                val_perm = folds[val_fold_idx]
                train_perm = list(
                    chain.from_iterable(
                        folds[:val_fold_idx] + folds[val_fold_idx + 1 :]
                    )
                )

                # Map permuted indices → molecule IDs → structure IDs
                val_mol_ids = [mol_ids[i] for i in val_perm]
                train_mol_ids = [mol_ids[i] for i in train_perm]

                train_structure_ids = self.dataset.get_structure_ids_for_mol(
                    train_mol_ids
                )
                val_structure_ids = self.dataset.get_structure_ids_for_mol(val_mol_ids)

                print(f"lables per class Train set{self.dataset.regression_masks[train_structure_ids,:].sum(dim= 0)}")

                print(f"lables per class Vals set{self.dataset.regression_masks[val_structure_ids,:].sum(dim= 0)}")


                yield train_structure_ids, val_structure_ids, f"Split_{i_repeat}-Fold_{val_fold_idx}"

    def _bemis_murcko_scaffold_splitting(self):
        raise NotImplementedError

        # Probably too complicated to implement

    def get_split(
        self, split_config: SplitConfig
    ) -> Iterator[tuple[list[int], list[int], str]]:
        """
        Unified entry‑point.  Yields (train_ids, val_ids) tuples according to
        the strategy described by *split_config*.
        """
        if split_config.strategy is SplitStrategy.SINGLE:
            yield self._single_split(
                split_config.train_val_ratios, split_config.shuffle
            )

        elif split_config.strategy is SplitStrategy.REPEATED_CV:
            yield from self._repeated_cv(
                N_repeats=split_config.N_repeats,
                N_splits=split_config.N_folds,
                shuffle=split_config.shuffle,
            )
        else:  # should never happen
            raise ValueError(f"Unknown split strategy: {split_config.strategy}")

    def visualise_splitting(self, N_repeats=5, N_splits=5):
        splits = list(self._repeated_cv(N_repeats, N_splits, shuffle=True))

        # ---------- build a 2‑D matrix: rows = splits×folds, cols = samples ------------
        N_structures = len(self.dataset.structure_ids)
        matrix = np.zeros((N_repeats * N_splits, N_structures))

        for row_idx, (train_idx, val_idx) in enumerate(splits):
            for col in val_idx:  # mark validation samples as 1
                matrix[row_idx, col] = 1

        # ---------- plot ----------
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="none"
        )  # default colormap

        # y‑labels: "Split i  Fold j"
        y_positions = []
        y_labels = []
        for rep in range(N_repeats):
            for fold in range(N_splits):
                idx = rep * N_splits + fold
                y_positions.append(idx)
                y_labels.append(f"Split {rep + 1}  Fold {fold + 1}")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Sample index")
        ax.set_title(f"Repeated {N_splits}-fold CV  ({N_repeats} repeats)")

        plt.tight_layout()
        return fig

    def general_split(self, split_ratios, shuffle):

        # 1. Get the set of mol ids
        mol_ids = self.dataset.get_all_mol_ids()
        N_mols = len(mol_ids)

        # 2 Get random permutation if shuffle true

        if shuffle:
            # perm =  list that contains randomly shuffeld indices
            rng = np.random.default_rng(seed = 1)
            perm = rng.permutation(N_mols).tolist()
        else:
            perm = list(range(N_mols))


        # Split the mol ids

        splitting_indices = compute_splits(N_mols, split_ratios)

        index_per_slice = []

        for splitting_slice in splitting_indices:


            split_perm = perm[splitting_slice.start : splitting_slice.stop]

            split_mol_ids = [mol_ids[i] for i in split_perm]
            structure_ids = self.dataset.get_structure_ids_for_mol(split_mol_ids)

            index_per_slice.append(structure_ids)

        return index_per_slice

    def materialise_dataset(self):
        pass
