"""Chirality class taxonomy shared by every pipeline stage.

Class label order follows the paper (Peng et al., AI Chemistry 2025):
``achiral=0, central=1, axial=2, helical=3, planar=4``.
"""

from __future__ import annotations

CLASS_ORDER: tuple[str, ...] = ("achiral", "central", "axial", "helical", "planar")
CLASS_TO_LABEL: dict[str, int] = {name: i for i, name in enumerate(CLASS_ORDER)}

# Normalise alternative spellings found in the raw data to the canonical names.
SYNONYMS: dict[str, str] = {"center": "central"}


def normalize_class(name: str) -> str:
    """Map a raw chiral-type string to a canonical class name.

    Raises ``ValueError`` for an unrecognised class so that silent mislabelling
    cannot happen.
    """
    canonical = SYNONYMS.get(name, name)
    if canonical not in CLASS_TO_LABEL:
        raise ValueError(
            f"Unknown chiral class {name!r} (normalised {canonical!r}); "
            f"expected one of {CLASS_ORDER}"
        )
    return canonical
