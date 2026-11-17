"""The named element-set presets resolve to the concrete element sets; the d-
and f-block metals."""

from __future__ import annotations

import pytest

from remedi_data.chemistry.elements import (
    D_BLOCK_ATOMIC_NUMBERS,
    F_BLOCK_ATOMIC_NUMBERS,
    MACE_OFF_ELEMENTS,
    MACE_POLAR_ELEMENTS,
    ElementSet,
    allowed_element_symbols,
    element_block,
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


@pytest.mark.parametrize(
    ("atomic_number", "block"),
    [
        (26, "3d"),  # Fe
        (30, "3d"),  # Zn
        (46, "4d"),  # Pd
        (57, "5d"),  # La: group 3
        (77, "5d"),  # Ir
        (58, "4f"),  # Ce
        (71, "4f"),  # Lu
        (89, "6d"),  # Ac: group 3
        (92, "5f"),  # U
        (103, "5f"),  # Lr
        (104, "6d"),  # Rf
        (6, "other"),
        (31, "other"),  # Ga
    ],
)
def test_element_block_rows(atomic_number: int, block: str) -> None:
    assert element_block(atomic_number) == block


def test_the_metal_sets_are_groups_3_to_12_and_the_f_block() -> None:
    assert D_BLOCK_ATOMIC_NUMBERS == frozenset(
        [*range(21, 31), *range(39, 49), 57, *range(72, 81), 89, *range(104, 113)]
    )
    assert F_BLOCK_ATOMIC_NUMBERS == frozenset([*range(58, 72), *range(90, 104)])
    assert not D_BLOCK_ATOMIC_NUMBERS & F_BLOCK_ATOMIC_NUMBERS
