"""The named element-set presets resolve to the concrete element sets."""

from __future__ import annotations

from remedi.data_handling.chemistry.elements import (
    MACE_OFF_ELEMENTS,
    MACE_POLAR_ELEMENTS,
    ElementSet,
    allowed_element_symbols,
)


def test_presets_map_to_concrete_sets() -> None:
    assert allowed_element_symbols(ElementSet.mace_off) is MACE_OFF_ELEMENTS
    assert allowed_element_symbols(ElementSet.mace_polar) is MACE_POLAR_ELEMENTS


def test_an_explicit_list_and_none_resolve() -> None:
    assert allowed_element_symbols(["C", "H"]) == frozenset({"C", "H"})
    assert allowed_element_symbols(None) is None


def test_mace_off_is_the_organic_drug_subset() -> None:
    assert MACE_OFF_ELEMENTS == {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def test_mace_polar_is_a_strict_superset_of_mace_off() -> None:
    assert MACE_OFF_ELEMENTS < MACE_POLAR_ELEMENTS
