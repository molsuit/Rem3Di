"""Deterministic scaffold splitter + Split-code helper.

`deepchem_scaffold_split` is the salvaged eval001 piece; it must stay
deterministic (no seed) and partition every index exactly once so the codes
written into the dataset `split` column are well-defined.
"""

from __future__ import annotations

import numpy as np

from remedi.data_handling.dataset.tasks import Split
from remedi.data_handling.dataset_creation.splits import (
    deepchem_scaffold_split,
    random_train_val_test_split,
    split_codes,
)

_SMILES = [
    "CCO",
    "CCN",
    "CCC",
    "c1ccccc1",
    "c1ccccc1C",
    "c1ccccc1O",
    "c1ccncc1",
    "c1ccncc1C",
    "C1CCCCC1",
    "C1CCCCC1O",
    "CC(=O)O",
    "CC(=O)N",
    "CCOC(=O)C",
    "c1ccc2ccccc2c1",
    "c1ccc2ccccc2c1C",
    "INVALID_SMILES",
    "O=C(O)c1ccccc1",
    "O=C(O)c1ccccc1N",
]


def _assert_partition(parts, n: int) -> None:
    allidx = np.concatenate(parts)
    np.testing.assert_array_equal(np.sort(allidx), np.arange(n))


def test_scaffold_split_deterministic() -> None:
    a = deepchem_scaffold_split(_SMILES)
    b = deepchem_scaffold_split(_SMILES)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_scaffold_split_partitions_every_index() -> None:
    train, valid, test = deepchem_scaffold_split(_SMILES)
    _assert_partition((train, valid, test), len(_SMILES))
    assert len(train) >= len(valid) and len(train) >= len(test)


def test_random_split_partitions_and_is_seeded() -> None:
    n = 40
    a = random_train_val_test_split(n, seed=7)
    b = random_train_val_test_split(n, seed=7)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    _assert_partition(a, n)


def test_split_codes_dtype_and_assignment() -> None:
    n = len(_SMILES)
    train, valid, test = deepchem_scaffold_split(_SMILES)
    codes = split_codes(n, train, valid, test)
    assert codes.dtype == np.uint8
    np.testing.assert_array_equal(codes[train], Split.train.value)
    np.testing.assert_array_equal(codes[valid], Split.valid.value)
    np.testing.assert_array_equal(codes[test], Split.test.value)
    assert (codes != Split.unassigned.value).all()
