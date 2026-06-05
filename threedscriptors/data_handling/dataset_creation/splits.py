"""Literature train/valid/test splitters, materialized into the dataset.

`deepchem_scaffold_split` is the genuinely valuable piece salvaged from the
junior eval001 panel: a faithful, deterministic re-implementation of
``deepchem.splits.ScaffoldSplitter.split`` (no 300 MB deepchem / TF dep). It
matches the MolCLR / Uni-Mol / GraphMVP / GEM / MoLFormer-XL scaffold-split
convention. Splits are computed once at ingest from the canonical SMILES that
go into the zarr and stored in the per-structure ``split`` column.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from threedscriptors.data_handling.dataset.tasks import Split


def deepchem_scaffold_split(
    smiles: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    include_chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical DeepChem ScaffoldSplitter algorithm — deterministic.

    Fully determined by the SMILES list and the train/val ratios (no seed:
    multiple seeds only vary model init in the literature convention).
    ``include_chirality=True`` matches Uni-Mol's modern consensus.

    Algorithm: Bemis-Murcko scaffold per SMILES; group indices by scaffold;
    sort groups by (size, first-index) descending (the first-index tiebreaker
    reproduces DeepChem / Hu et al. exactly); greedily fill train, then valid,
    then test by the cutoffs.
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds[f"INVALID:{smi}"].append(i)
            continue
        scaffolds[
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=include_chirality
            )
        ].append(i)

    scaffold_sets = sorted(
        scaffolds.values(), key=lambda x: (len(x), x[0]), reverse=True
    )
    n = len(smiles)
    train_cutoff = train_frac * n
    valid_cutoff = (train_frac + val_frac) * n

    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []
    for inds in scaffold_sets:
        if len(train_idx) + len(inds) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(inds) > valid_cutoff:
                test_idx.extend(inds)
            else:
                valid_idx.extend(inds)
        else:
            train_idx.extend(inds)

    return (
        np.sort(np.asarray(train_idx, dtype=int)),
        np.sort(np.asarray(valid_idx, dtype=int)),
        np.sort(np.asarray(test_idx, dtype=int)),
    )


def random_train_val_test_split(
    n: int,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded random split — the non-scaffold alternative."""
    idx = np.arange(n)
    test_frac = 1.0 - train_frac - val_frac
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=seed, shuffle=True
    )
    val_relative = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=seed + 10_000,
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def stratified_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware, class-stratified train/valid/test split.

    Each *group* (e.g. all conformers / enantiomers sharing a non-isomeric
    SMILES) is assigned whole to exactly one partition so the same molecular
    graph never leaks across folds. Within that constraint the partition class
    proportions track the global ones, so rare classes (e.g. the 37 ``helical``
    chiral_cat molecules) land in train, valid and test rather than collapsing
    into one fold.

    Implementation: assign one label per group (the group's modal label), order
    groups by class so each class is dealt round-robin across the three
    partitions by their target fractions, and break ties deterministically with
    a seeded shuffle *within* each (class) bucket. Returns sorted index arrays
    over the original rows.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    n = labels.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty.copy(), empty.copy()

    rng = np.random.default_rng(seed)

    # One representative label per group (modal label within the group).
    group_to_rows: dict[object, list[int]] = defaultdict(list)
    for i in range(n):
        group_to_rows[groups[i].item() if hasattr(groups[i], "item") else groups[i]].append(i)

    group_label: dict[object, int] = {}
    for g, rows in group_to_rows.items():
        vals, counts = np.unique(labels[rows], return_counts=True)
        group_label[g] = int(vals[int(np.argmax(counts))])

    test_frac = 1.0 - train_frac - val_frac
    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []

    # Deal groups per class so every partition gets a proportional share of
    # each class. Cumulative-fraction assignment keeps the smallest classes
    # spread across all three folds instead of rounding them all into train.
    classes = sorted({group_label[g] for g in group_to_rows})
    for cls in classes:
        cls_groups = [g for g in group_to_rows if group_label[g] == cls]
        cls_groups.sort(
            key=lambda g: (len(group_to_rows[g]), g if isinstance(g, str) else str(g))
        )
        rng.shuffle(cls_groups)
        for k, g in enumerate(cls_groups):
            pos = (k + 0.5) / len(cls_groups)
            if pos < train_frac:
                train_idx.extend(group_to_rows[g])
            elif pos < train_frac + val_frac or test_frac <= 0.0:
                valid_idx.extend(group_to_rows[g])
            else:
                test_idx.extend(group_to_rows[g])

    return (
        np.sort(np.asarray(train_idx, dtype=int)),
        np.sort(np.asarray(valid_idx, dtype=int)),
        np.sort(np.asarray(test_idx, dtype=int)),
    )


def split_codes(
    n: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Build the per-structure uint8 Split-code array the writer expects.

    Any index not in a partition stays ``Split.unassigned`` (defensive; the
    splitters above partition every index).
    """
    codes = np.full(n, Split.unassigned.value, dtype="u1")
    codes[np.asarray(train_idx, dtype=int)] = Split.train.value
    codes[np.asarray(valid_idx, dtype=int)] = Split.valid.value
    codes[np.asarray(test_idx, dtype=int)] = Split.test.value
    return codes
