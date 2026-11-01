"""The geometry guards every structure passes before it reaches a model."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict

from remedi.data_handling.chemistry.elements import (
    ElementSet,
    allowed_element_symbols,
)


class GeometryLimits(BaseModel):
    """Size, element, hydrogen-coverage and atom-overlap limits for one structure.

    ``elements`` is a named :class:`ElementSet` preset or an explicit list of
    element symbols; ``None`` disables the element gate (a curated source such
    as tmQM may legitimately carry anything). ``min_interatomic_distance`` in
    Angstrom guards against overlapping atoms, on which MACE divides by a
    near-zero distance and emits NaN embeddings; ~0.5 A is safely below any
    real bond length.
    """

    model_config = ConfigDict(extra="forbid")

    max_atoms: int | None = None
    elements: ElementSet | list[str] | None = None
    reject_zero_hydrogen: bool = False
    min_hydrogen_heavy_ratio: float = 0.0
    min_interatomic_distance: float | None = None

    def allowed_element_symbols(self) -> frozenset[str] | None:
        """Resolve ``elements`` to a set of symbols, or ``None`` if ungated."""
        return allowed_element_symbols(self.elements)


def minimum_interatomic_distance(positions: np.ndarray) -> float:
    """Smallest distance between any two atoms, in Angstrom."""
    if len(positions) < 2:
        return float("inf")
    difference = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(difference, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def geometry_violations(
    atoms: Atoms,
    limits: GeometryLimits,
    allowed_symbols: frozenset[str] | None = None,
) -> list[str]:
    """The names of the limits one structure violates; empty means it passes.

    The names (``max_atoms``, ``no_heavy_atom``, ``zero_hydrogen``,
    ``hydrogen_heavy_ratio``, ``element_gate``, ``min_interatomic_distance``)
    double as ``counts.dropped`` keys. ``allowed_symbols`` lets a caller that
    checks many structures resolve the element set once; by default it is
    resolved from ``limits``.
    """
    if allowed_symbols is None:
        allowed_symbols = limits.allowed_element_symbols()
    violations: list[str] = []
    if limits.max_atoms is not None and len(atoms) > limits.max_atoms:
        violations.append("max_atoms")
    atomic_numbers = atoms.get_atomic_numbers()
    hydrogen_count = int((atomic_numbers == 1).sum())
    heavy_count = int((atomic_numbers > 1).sum())
    if heavy_count == 0:
        violations.append("no_heavy_atom")
    elif limits.reject_zero_hydrogen and hydrogen_count == 0:
        violations.append("zero_hydrogen")
    elif (
        limits.min_hydrogen_heavy_ratio > 0.0
        and (hydrogen_count / heavy_count) < limits.min_hydrogen_heavy_ratio
    ):
        violations.append("hydrogen_heavy_ratio")
    if allowed_symbols is not None and not set(atoms.get_chemical_symbols()).issubset(
        allowed_symbols
    ):
        violations.append("element_gate")
    if (
        limits.min_interatomic_distance is not None
        and minimum_interatomic_distance(atoms.get_positions())
        < limits.min_interatomic_distance
    ):
        violations.append("min_interatomic_distance")
    return violations
