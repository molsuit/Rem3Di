"""The element sets the filters and geometry limits gate on."""

from __future__ import annotations

from enum import StrEnum


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
