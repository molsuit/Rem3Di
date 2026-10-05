"""CXSMILES enhanced stereo -> plain SMILES (``chemistry/stereo_groups.py``)."""

from __future__ import annotations

import pytest

from remedi.data_handling.chemistry.stereo_groups import (
    has_or_stereo_group,
    unspecify_relative_stereo,
)

# From the ASAP antiviral sets: one OR group over two centres, one AND group
# over two centres, one absolute centre.
OR_GROUP = "C[C@H](C(=O)O)[C@@H](NC(=O)C1=NC=NC2=C1C=CN2)C1=CC=C(Br)C=C1 |o1:1,5|"
AND_GROUP = "CC(=O)N1C[C@@]2(CC[C@H]1C)NC(=O)N(C1=CN=CC3=CC=CC=C13)C2=O |&1:5,8|"
ABSOLUTE = "COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1 |a:16|"


@pytest.mark.parametrize(
    ("smiles", "expected", "unspecified_centres"),
    [
        (OR_GROUP, None, 2),
        (AND_GROUP, None, 2),
        (ABSOLUTE, "COc1cccc(Cl)c1NC(=O)N1CCC[C@H](C(N)=O)C1", 0),
        # Only the grouped centre is unspecified; the other survives.
        ("C[C@H](N)[C@@H](C)O |o1:1|", "CC(N)[C@@H](C)O", 1),
        ("C[C@@H](N)C(=O)O", "C[C@@H](N)C(=O)O", 0),
    ],
    ids=["or group", "and group", "absolute", "partly grouped", "plain smiles"],
)
def test_grouped_centres_lose_their_drawn_configuration(
    smiles: str, expected: str | None, unspecified_centres: int
) -> None:
    result = unspecify_relative_stereo(smiles)
    assert result is not None
    assert "|" not in result.smiles
    if expected is None:  # every centre sat in the group
        assert "@" not in result.smiles
    else:
        assert result.smiles == expected
    assert result.unspecified_centres == unspecified_centres
    assert unspecify_relative_stereo("not a molecule") is None


def test_only_an_or_group_marks_an_unassigned_enantiomer() -> None:
    assert has_or_stereo_group(OR_GROUP) is True
    assert has_or_stereo_group(AND_GROUP) is False
    assert has_or_stereo_group(ABSOLUTE) is False
    assert has_or_stereo_group("C[C@@H](N)C(=O)O") is False
    assert has_or_stereo_group("not a molecule") is False
