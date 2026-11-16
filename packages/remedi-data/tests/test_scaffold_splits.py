"""The splitters in ``chemistry/splits.py``.

`deepchem_scaffold_split` is the salvaged eval001 piece; it must stay
deterministic (no seed) and partition every index exactly once so the codes
written into the dataset `split` column are well-defined.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit.Chem.Scaffolds import MurckoScaffold

from remedi_data.chemistry.splits import (
    deepchem_scaffold_split,
    fixed_test_seeded_split_columns,
    murcko_scaffold,
    scaffold_groups,
    scaffold_test_mask,
    seeded_scaffold_train_valid_split,
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


def test_scaffold_split_is_deterministic_and_partitions_every_index() -> None:
    parts = deepchem_scaffold_split(_SMILES)
    _assert_partition(parts, len(_SMILES))
    train, valid, test = parts
    assert len(train) >= len(valid) and len(train) >= len(test)
    for x, y in zip(parts, deepchem_scaffold_split(_SMILES), strict=True):
        np.testing.assert_array_equal(x, y)


# ------------------------------------------- seeded scaffold train / valid


# 60 molecules over 12 scaffolds: n-alkyl chains on six ring systems, plus
# acyclic ones that all share the empty scaffold.
_RINGS = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1ccsc1", "C1CCNCC1"]
_MANY_SMILES = [
    f"{'C' * length}{ring}" for ring in _RINGS for length in range(1, 9)
] + ["C" * length + "O" for length in range(1, 13)]


def test_seeded_scaffold_split_keeps_scaffolds_whole_and_is_seeded() -> None:
    first = seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=1)
    _assert_partition(first, len(_MANY_SMILES))
    valid_set = set(first[1].tolist())
    assert valid_set
    for positions in scaffold_groups(_MANY_SMILES).values():
        assert len({position in valid_set for position in positions}) == 1
    again = seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=1)
    for x, y in zip(first, again, strict=True):
        np.testing.assert_array_equal(x, y)
    others = [
        seeded_scaffold_train_valid_split(_MANY_SMILES, 0.2, seed=seed)[1]
        for seed in range(2, 8)
    ]
    assert any(not np.array_equal(first[1], other) for other in others)
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


def test_fixed_test_columns_reject_bad_inputs() -> None:
    is_test = np.zeros(len(_MANY_SMILES), dtype=bool)
    for nothing_or_everything in (is_test, ~is_test):
        with pytest.raises(ValueError, match="neither empty nor everything"):
            fixed_test_seeded_split_columns(
                _MANY_SMILES, nothing_or_everything, [1], 0.2
            )
    is_test[0] = True
    with pytest.raises(ValueError, match="unique"):
        fixed_test_seeded_split_columns(_MANY_SMILES, is_test, [1, 1], 0.2)
    with pytest.raises(ValueError, match="shape"):
        fixed_test_seeded_split_columns(_MANY_SMILES, is_test[:3], [1], 0.2)


def test_an_rdkit_error_in_the_scaffold_is_an_invalid_scaffold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RDKit can raise inside ``GetScaffoldForMol`` on a SMILES that parses."""
    real_scaffold = MurckoScaffold.GetScaffoldForMol

    def raising_on_toluene(mol):
        if mol.GetNumAtoms() == 7:
            raise RuntimeError("Pre-condition Violation")
        return real_scaffold(mol)

    monkeypatch.setattr(MurckoScaffold, "GetScaffoldForMol", raising_on_toluene)
    assert murcko_scaffold("Cc1ccccc1") is None
    assert murcko_scaffold("CCc1ccccc1") == "c1ccccc1"
    assert scaffold_groups(["Cc1ccccc1", "CCc1ccccc1"]) == {
        "INVALID:Cc1ccccc1": [0],
        "c1ccccc1": [1],
    }


@pytest.mark.parametrize(
    ("smiles", "specific_chiral", "specific", "generic"),
    [
        (
            "C1CC[C@H]2CCCC[C@@H]2C1",
            "C1CC[C@H]2CCCC[C@@H]2C1",
            "C1CCC2CCCCC2C1",
            "C1CCC2CCCCC2C1",
        ),
        (
            "O=C(Nc1ccccc1)c1ccncc1",
            "O=C(Nc1ccccc1)c1ccncc1",
            "O=C(Nc1ccccc1)c1ccncc1",
            "CC(CC1CCCCC1)C1CCCCC1",
        ),
        (
            "c1ccc(-c2ccccn2)cc1",
            "c1ccc(-c2ccccn2)cc1",
            "c1ccc(-c2ccccn2)cc1",
            "C1CCC(C2CCCCC2)CC1",
        ),
        ("CCO", "", "", ""),
    ],
)
def test_murcko_scaffold_strings_are_pinned(
    smiles: str, specific_chiral: str, specific: str, generic: str
) -> None:
    """The scaffold strings splits, selection and the dataset analyses group by."""
    assert murcko_scaffold(smiles, include_chirality=True) == specific_chiral
    assert murcko_scaffold(smiles, include_chirality=False) == specific
    assert murcko_scaffold(smiles, include_chirality=True, generic=True) == generic
    assert murcko_scaffold(smiles, include_chirality=False, generic=True) == generic


def test_an_unparsable_smiles_has_no_scaffold() -> None:
    assert murcko_scaffold("C1CC") is None
    assert murcko_scaffold("C1CC", generic=True) is None
