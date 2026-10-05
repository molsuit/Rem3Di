"""The splitters in ``chemistry/splits.py`` + the Split-code helper.

`deepchem_scaffold_split` is the salvaged eval001 piece; it must stay
deterministic (no seed) and partition every index exactly once so the codes
written into the dataset `split` column are well-defined.

`stratified_group_split` is what the ChiralCat preparer freezes its partition
with: it must keep a group (a constitution, hence an enantiomer pair) whole and
still put every class in every fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from remedi.data_handling.chemistry.splits import (
    deepchem_scaffold_split,
    fixed_test_seeded_split_columns,
    random_train_val_test_split,
    scaffold_groups,
    scaffold_test_mask,
    seeded_scaffold_train_valid_split,
    split_codes,
    stratified_group_split,
)
from remedi.data_handling.dataset.tasks import Split

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


# ------------------------------------------------- stratified group splitting


def test_stratified_group_split_keeps_groups_whole_and_spreads_classes() -> None:
    # 30 groups per class over 3 classes; one row per group.
    labels = np.repeat([0, 1, 2], 30)
    groups = np.array([f"g{i}" for i in range(len(labels))], dtype=object)
    train, valid, test = stratified_group_split(labels, groups, 0.6, 0.2, seed=0)

    # Partition every row exactly once, no overlap.
    assert sorted([*train, *valid, *test]) == list(range(len(labels)))
    assert set(train).isdisjoint(valid)
    assert set(train).isdisjoint(test)
    assert set(valid).isdisjoint(test)
    # Every class appears in every partition.
    for partition in (train, valid, test):
        assert set(labels[partition].tolist()) == {0, 1, 2}


def test_stratified_group_split_no_group_leakage() -> None:
    # Two rows per group; the pair must land in the same partition.
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    groups = np.array(
        ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"], dtype=object
    )
    train, valid, test = stratified_group_split(labels, groups, 0.5, 0.25, seed=1)
    partition_by_row: dict[int, str] = {}
    for name, partition in (("train", train), ("valid", valid), ("test", test)):
        for row in partition:
            partition_by_row[row] = name
    for group in set(groups):
        rows = [index for index, value in enumerate(groups) if value == group]
        assert len({partition_by_row[row] for row in rows}) == 1, (
            f"group {group} leaked across partitions"
        )


# ------------------------------------------- seeded scaffold train / valid


# 60 molecules over 12 scaffolds: n-alkyl chains on six ring systems, plus
# acyclic ones that all share the empty scaffold.
_RINGS = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1ccsc1", "C1CCNCC1"]
_MANY_SMILES = [
    f"{'C' * length}{ring}" for ring in _RINGS for length in range(1, 9)
] + ["C" * length + "O" for length in range(1, 13)]


def test_seeded_scaffold_split_partitions_and_keeps_scaffolds_whole() -> None:
    train, valid = seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=3)
    _assert_partition((train, valid), len(_MANY_SMILES))
    assert len(valid) > 0
    groups = scaffold_groups(_MANY_SMILES)
    valid_set = set(valid.tolist())
    for positions in groups.values():
        assert len({position in valid_set for position in positions}) == 1


def test_seeded_scaffold_split_is_seeded() -> None:
    first = seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=1)
    again = seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=1)
    for x, y in zip(first, again, strict=True):
        np.testing.assert_array_equal(x, y)
    others = [
        seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=seed)[1]
        for seed in range(2, 8)
    ]
    assert any(not np.array_equal(first[1], other) for other in others)


def test_seeded_scaffold_split_rejects_a_bad_fraction() -> None:
    with pytest.raises(ValueError, match="valid_fraction"):
        seeded_scaffold_train_valid_split(_MANY_SMILES, 1.0, seed=1)


def test_fixed_test_columns_keep_the_test_fold_and_alias_the_first_seed() -> None:
    is_test = np.zeros(len(_MANY_SMILES), dtype=bool)
    is_test[::7] = True
    columns = fixed_test_seeded_split_columns(
        _MANY_SMILES, is_test, seeds=[1, 2, 3], valid_fraction=0.2
    )
    assert list(columns) == ["split", "split__seed1", "split__seed2", "split__seed3"]
    assert columns["split"] == columns["split__seed1"]
    for values in columns.values():
        assert len(values) == len(_MANY_SMILES)
        assert [value == "test" for value in values] == is_test.tolist()
        assert set(values) == {"train", "valid", "test"}


def test_scaffold_test_mask_is_the_deepchem_test_part() -> None:
    _, _, test = deepchem_scaffold_split(_MANY_SMILES)
    mask = scaffold_test_mask(_MANY_SMILES)
    assert mask.dtype == bool
    np.testing.assert_array_equal(np.flatnonzero(mask), test)


def test_fixed_test_columns_reject_an_empty_or_total_test_fold() -> None:
    nothing = np.zeros(len(_MANY_SMILES), dtype=bool)
    with pytest.raises(ValueError, match="neither empty nor everything"):
        fixed_test_seeded_split_columns(_MANY_SMILES, nothing, [1], 0.2)
    with pytest.raises(ValueError, match="neither empty nor everything"):
        fixed_test_seeded_split_columns(_MANY_SMILES, ~nothing, [1], 0.2)


def test_fixed_test_columns_reject_bad_inputs() -> None:
    is_test = np.zeros(len(_MANY_SMILES), dtype=bool)
    is_test[0] = True
    with pytest.raises(ValueError, match="unique"):
        fixed_test_seeded_split_columns(_MANY_SMILES, is_test, [1, 1], 0.2)
    with pytest.raises(ValueError, match="shape"):
        fixed_test_seeded_split_columns(_MANY_SMILES, is_test[:3], [1], 0.2)
