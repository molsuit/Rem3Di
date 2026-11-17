"""Element sets: the presets the filters and geometry limits gate on, and the
d- and f-block metals (:func:`element_block`)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class ElementSet(StrEnum):
    """A named element-set preset.

    ``mace_off`` is the organic drug subset (H, C, N, O, F, P, S, Cl, Br, I).
    ``mace_polar`` covers atomic numbers 1..83, the MACE-POLAR coverage.
    """

    mace_off = "mace_off"
    mace_polar = "mace_polar"


#: MACE-OFF24 element coverage: the drug-like organic subset.
MACE_OFF_ELEMENTS: frozenset[str] = frozenset(
    {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
)

#: MACE-POLAR-1-M element coverage, atomic numbers 1..83 (H through Bi). Verified
#: 2026-05-03 by introspecting the model's ``atomic_numbers`` buffer.
MACE_POLAR_ELEMENTS: frozenset[str] = frozenset(
    (
        "H He "
        "Li Be B C N O F Ne "
        "Na Mg Al Si P S Cl Ar "
        "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
        "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
        "Cs Ba "
        "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
        "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi"
    ).split()
)

_ELEMENTS_BY_SET: dict[ElementSet, frozenset[str]] = {
    ElementSet.mace_off: MACE_OFF_ELEMENTS,
    ElementSet.mace_polar: MACE_POLAR_ELEMENTS,
}


def allowed_element_symbols(
    elements: ElementSet | list[str] | None,
) -> frozenset[str] | None:
    """Resolve a preset or an explicit symbol list; ``None`` means ungated."""
    if elements is None:
        return None
    if isinstance(elements, ElementSet):
        return _ELEMENTS_BY_SET[elements]
    return frozenset(elements)


#: The d- and f-block rows :func:`element_block` names; everything else is
#: ``other``.
ElementBlock = Literal["3d", "4d", "5d", "6d", "4f", "5f", "other"]

_LAST_ATOMIC_NUMBER = 118


def element_block(atomic_number: int) -> ElementBlock:
    """The d or f sub-shell row of an element, ``other`` for the s and p blocks.

    Group 3 is Sc, Y, La, Ac (La and Ac are d-block), so the f-block rows are
    Ce-Lu (``4f``) and Th-Lr (``5f``).
    """
    if 21 <= atomic_number <= 30:
        return "3d"
    if 39 <= atomic_number <= 48:
        return "4d"
    if atomic_number == 57 or 72 <= atomic_number <= 80:
        return "5d"
    if atomic_number == 89 or 104 <= atomic_number <= 112:
        return "6d"
    if 58 <= atomic_number <= 71:
        return "4f"
    if 90 <= atomic_number <= 103:
        return "5f"
    return "other"


#: Transition metals: the d block, groups 3-12 (Sc-Zn, Y-Cd, La and Hf-Hg, Ac
#: and Rf-Cn).
D_BLOCK_ATOMIC_NUMBERS: frozenset[int] = frozenset(
    number
    for number in range(1, _LAST_ATOMIC_NUMBER + 1)
    if element_block(number) in ("3d", "4d", "5d", "6d")
)

#: Lanthanides and actinides without La and Ac: Ce-Lu and Th-Lr.
F_BLOCK_ATOMIC_NUMBERS: frozenset[int] = frozenset(
    number
    for number in range(1, _LAST_ATOMIC_NUMBER + 1)
    if element_block(number) in ("4f", "5f")
)
