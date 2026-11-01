"""Literature train/valid/test splitters, materialized into the dataset.

The SMILES preparers freeze a fixed test fold (:func:`scaffold_test_mask` or a
shipped one) plus seeded scaffold train/valid partitions
(:func:`fixed_test_seeded_split_columns`).

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


def scaffold_groups(
    smiles: list[str], include_chirality: bool = True
) -> dict[str, list[int]]:
    """Bemis-Murcko scaffold -> the positions of ``smiles`` that share it.

    Groups are keyed in first-seen order; an unparseable SMILES is its own group.
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
    return scaffolds


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
    scaffolds = scaffold_groups(smiles, include_chirality=include_chirality)
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


def seeded_scaffold_train_valid_split(
    smiles: list[str],
    valid_fraction: float,
    seed: int,
    include_chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded scaffold partition of ``smiles`` into train and valid.

    The chemprop "balanced" scaffold order: scaffold groups larger than half
    the valid target go first, then the rest, each part shuffled by ``seed``;
    groups fill train up to ``1 - valid_fraction`` of the rows and the rest go
    to valid. Putting the large groups first keeps one big scaffold from
    swallowing the valid fold. A scaffold group is never split.
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError(f"valid_fraction must be in (0, 1), got {valid_fraction}")
    groups = list(scaffold_groups(smiles, include_chirality=include_chirality).values())
    valid_target = valid_fraction * len(smiles)
    large = [group for group in groups if len(group) > valid_target / 2]
    small = [group for group in groups if len(group) <= valid_target / 2]
    generator = np.random.default_rng(seed)
    ordered = [large[i] for i in generator.permutation(len(large))] + [
        small[i] for i in generator.permutation(len(small))
    ]

    train_cutoff = (1.0 - valid_fraction) * len(smiles)
    train_idx: list[int] = []
    valid_idx: list[int] = []
    for group in ordered:
        if len(train_idx) + len(group) > train_cutoff:
            valid_idx.extend(group)
        else:
            train_idx.extend(group)
    return (
        np.sort(np.asarray(train_idx, dtype=int)),
        np.sort(np.asarray(valid_idx, dtype=int)),
    )


def fixed_test_seeded_split_columns(
    smiles: list[str],
    is_test: np.ndarray,
    seeds: list[int],
    valid_fraction: float,
    include_chirality: bool = True,
) -> dict[str, list[str]]:
    """Split columns with a fixed test fold and one seeded train/valid per seed.

    ``is_test`` marks the fixed test rows; every other row is partitioned by
    :func:`seeded_scaffold_train_valid_split` once per seed, with
    ``valid_fraction`` relative to the non-test rows. Returns
    ``{"split": …, "split__seed<s>": …}`` with ``split`` aliasing the first
    seed, each a ``train``/``valid``/``test`` value per row (§5c).
    """
    is_test = np.asarray(is_test, dtype=bool)
    if is_test.shape != (len(smiles),):
        raise ValueError(f"is_test has shape {is_test.shape} for {len(smiles)} rows")
    if not is_test.any() or is_test.all():
        raise ValueError(
            f"the fixed test fold holds {int(is_test.sum())} of {len(smiles)} rows; "
            "it must be neither empty nor everything"
        )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be non-empty and unique, got {seeds}")
    pool = np.flatnonzero(~is_test)
    pool_smiles = [smiles[position] for position in pool]
    columns: dict[str, list[str]] = {}
    for seed in seeds:
        values = np.full(len(smiles), "test", dtype=object)
        train_idx, valid_idx = seeded_scaffold_train_valid_split(
            pool_smiles, valid_fraction, seed, include_chirality=include_chirality
        )
        values[pool[train_idx]] = "train"
        values[pool[valid_idx]] = "valid"
        columns[f"split__seed{seed}"] = [str(value) for value in values]
    return {"split": list(columns[f"split__seed{seeds[0]}"]), **columns}


def scaffold_test_mask(
    smiles: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    include_chirality: bool = True,
) -> np.ndarray:
    """The test part of :func:`deepchem_scaffold_split` as a boolean row mask."""
    _, _, test_idx = deepchem_scaffold_split(
        smiles, train_frac, val_frac, include_chirality=include_chirality
    )
    is_test = np.zeros(len(smiles), dtype=bool)
    is_test[test_idx] = True
    return is_test
