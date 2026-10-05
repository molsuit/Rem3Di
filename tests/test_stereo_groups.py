"""CXSMILES enhanced stereo -> plain SMILES (``chemistry/stereo_groups.py``)."""

from __future__ import annotations

from remedi.data_handling.chemistry.stereo_groups import unspecify_relative_stereo

# From the ASAP antiviral sets: one OR group over two centres, one AND group
# over two centres, one absolute centre.
OR_GROUP = "C[C@H](C(=O)O)[C@@H](NC(=O)C1=NC=NC2=C1C=CN2)C1=CC=C(Br)C=C1 |o1:1,5|"
AND_GROUP = "CC(=O)N1C[C@@]2(CC[C@H]1C)NC(=O)N(C1=CN=CC3=CC=CC=C13)C2=O |&1:5,8|"
ABSOLUTE = "COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1 |a:16|"


def test_or_and_and_groups_lose_their_drawn_configuration() -> None:
    for smiles in (OR_GROUP, AND_GROUP):
        result = unspecify_relative_stereo(smiles)
        assert result is not None
        assert "@" not in result.smiles
        assert "|" not in result.smiles
        assert result.unspecified_centres == 2


def test_an_absolute_centre_keeps_its_configuration() -> None:
    result = unspecify_relative_stereo(ABSOLUTE)
    assert result is not None
    assert result.smiles == "COc1cccc(Cl)c1NC(=O)N1CCC[C@H](C(N)=O)C1"
    assert result.unspecified_centres == 0


def test_only_the_grouped_centres_are_unspecified() -> None:
    result = unspecify_relative_stereo("C[C@H](N)[C@@H](C)O |o1:1|")
    assert result is not None
    assert result.smiles == "CC(N)[C@@H](C)O"  # the ungrouped centre survives
    assert result.unspecified_centres == 1


def test_a_plain_smiles_passes_through_canonicalised() -> None:
    result = unspecify_relative_stereo("C[C@@H](N)C(=O)O")
    assert result is not None
    assert result.smiles == "C[C@@H](N)C(=O)O"
    assert result.unspecified_centres == 0


def test_an_unparseable_smiles_gives_none() -> None:
    assert unspecify_relative_stereo("not a molecule") is None
