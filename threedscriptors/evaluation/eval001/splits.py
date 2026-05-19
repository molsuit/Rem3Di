from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def _scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"INVALID:{smiles}"
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True)
    return Chem.MolToSmiles(core, canonical=True)


def random_train_val_test_split(
    n: int,
    seed: int,
    y: np.ndarray | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    stratify = None
    if y is not None and np.asarray(y).ndim == 1:
        yy = np.asarray(y)
        values, counts = np.unique(yy[~np.isnan(yy)], return_counts=True)
        if len(values) == 2 and np.all(counts >= 3):
            stratify = yy

    test_frac = 1.0 - train_frac - val_frac
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    stratify_train_val = None
    if stratify is not None:
        stratify_train_val = stratify[train_val_idx]
    val_relative = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=seed + 10_000,
        shuffle=True,
        stratify=stratify_train_val,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def scaffold_train_val_test_split(
    smiles: list[str],
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy custom scaffold splitter (Bemis-Murcko + seeded greedy bin-pack).
    Use `deepchem_scaffold_split` for publication-grade evaluation matching
    MolCLR / Uni-Mol / GraphMVP / MuMo convention."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[_scaffold_key(smi)].append(i)

    rng = np.random.default_rng(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    n = len(smiles)
    targets = np.array([train_frac, val_frac, 1.0 - train_frac - val_frac]) * n
    buckets: list[list[int]] = [[], [], []]
    sizes = np.zeros(3, dtype=float)
    for _, indices in items:
        ratios = sizes / np.maximum(targets, 1.0)
        bucket = int(np.argmin(ratios))
        buckets[bucket].extend(indices)
        sizes[bucket] += len(indices)

    return tuple(np.sort(np.asarray(b, dtype=int)) for b in buckets)  # type: ignore[return-value]


def deepchem_scaffold_split(
    smiles: list[str],
    seed: int = 0,  # ignored; kept for caller-interface compatibility
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    include_chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical DeepChem ScaffoldSplitter algorithm — deterministic.

    Matches the protocol used by MolCLR (Wang 2022 Nat MI), Uni-Mol (Zhou 2023
    ICLR), GraphMVP (Liu 2022 ICLR), GEM (Fang 2022 Nat MI), MoLFormer-XL (Ross
    2022 Nat MI). The split is fully determined by the smiles list and
    train/val/test ratios; seed has no effect on the split (multiple seeds
    only vary model initialisation, per the convention in those papers).

    **`include_chirality=True` matches Uni-Mol's modern consensus** (Zhou 2023
    ICLR §4.1, p. 13: "we reproduce [MolCLR] by considering [chirality]";
    the harder, more honest variant). Hu et al. 2020 (the original Bemis-Murcko
    pretraining-GNNs protocol) also defaults to True. Setting this to False
    matches MolCLR's original numbers but is now considered superseded.

    Faithful re-implementation of `deepchem.splits.ScaffoldSplitter.split` to
    avoid pulling in the 300+ MB deepchem dependency (and the
    tensorflow-on-python<3.12 constraint). Algorithm:
        1. Compute Bemis-Murcko scaffold per SMILES with chirality flag.
        2. Group indices by scaffold.
        3. Sort scaffold groups by (size, first-index) descending — the
           first-index tiebreaker reproduces DeepChem / Hu et al. exactly.
        4. Assign each scaffold group greedily to train, else val, else test,
           keeping <= train_frac * n in train, <= (train+val)_frac * n in
           train + val.
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds[f"INVALID:{smi}"].append(i)
            continue
        scaffolds[MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality)].append(i)

    # Tie-breaker: (size, first-index) descending — matches DeepChem / Hu et al.
    scaffold_sets = sorted(scaffolds.values(),
                            key=lambda x: (len(x), x[0]), reverse=True)
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

    return (np.sort(np.asarray(train_idx, dtype=int)),
            np.sort(np.asarray(valid_idx, dtype=int)),
            np.sort(np.asarray(test_idx, dtype=int)))

