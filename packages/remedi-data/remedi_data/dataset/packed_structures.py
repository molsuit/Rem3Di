"""``PackedStructures``: many structures as flat arrays, no per-structure objects.

The in-memory form the writer, the validator, the structure preparers and the
training data pass around. Structure ``i`` owns atoms ``ptr[i]:ptr[i + 1]`` of
``positions`` and ``atomic_numbers``; ``total_charge`` and ``multiplicity``, one
per structure, are there when the structures were read from a dataset (or are
about to be written to one) and ``None`` where only the frames are known. The
dtypes are the ones the zarr stores (float32 positions, charges and
multiplicities, uint8 atomic numbers), so whatever is checked in memory is
exactly what is written.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike

POSITION_DTYPE = np.float32
ATOMIC_NUMBER_DTYPE = np.uint8
#: ``total_charge`` and ``multiplicity``.
PER_STRUCTURE_DTYPE = np.float32


def gather_index(
    ptr: np.ndarray, structure_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The atom rows of ``structure_indices`` (in that order) and their new ``ptr``.

    Returns:
        ``(new_ptr, atom_index)``: ``atom_index`` lists the atoms of the chosen
        structures, so ``positions[atom_index]`` (or a zarr array's
        ``oindex[atom_index]``) holds them back to back, and ``new_ptr`` is
        their pointer, starting at 0.
    """
    structure_indices = np.asarray(structure_indices, dtype=np.int64)
    starts = ptr[structure_indices]
    counts = ptr[structure_indices + 1] - starts
    new_ptr = np.zeros(len(structure_indices) + 1, dtype=np.int64)
    np.cumsum(counts, out=new_ptr[1:])
    atom_index = np.repeat(starts - new_ptr[:-1], counts) + np.arange(
        new_ptr[-1], dtype=np.int64
    )
    return new_ptr, atom_index


def _per_structure(values: ArrayLike | None) -> np.ndarray | None:
    return None if values is None else np.ascontiguousarray(values, PER_STRUCTURE_DTYPE)


@dataclass(frozen=True)
class PackedStructures:
    """``positions`` ``(n_atoms, 3)`` float32, ``atomic_numbers`` ``(n_atoms,)`` uint8,
    ``ptr`` ``(n_structures + 1,)`` int64 starting at 0; optional
    ``total_charge`` and ``multiplicity`` ``(n_structures,)`` float32."""

    positions: np.ndarray
    atomic_numbers: np.ndarray
    ptr: np.ndarray
    total_charge: np.ndarray | None = None
    multiplicity: np.ndarray | None = None

    def __post_init__(self) -> None:
        ptr = self.ptr
        if ptr.ndim != 1 or len(ptr) == 0 or ptr[0] != 0:
            raise ValueError("ptr must be one-dimensional and start at 0")
        if np.any(np.diff(ptr) < 0):
            raise ValueError("ptr must be non-decreasing")
        if self.positions.shape != (int(ptr[-1]), 3):
            raise ValueError(
                f"positions has shape {self.positions.shape}, expected ({ptr[-1]}, 3)"
            )
        if self.atomic_numbers.shape != (int(ptr[-1]),):
            raise ValueError(
                f"atomic_numbers has shape {self.atomic_numbers.shape}, "
                f"expected ({ptr[-1]},)"
            )
        for name in ("total_charge", "multiplicity"):
            values = getattr(self, name)
            if values is not None and values.shape != (len(ptr) - 1,):
                raise ValueError(
                    f"{name} has shape {values.shape}, expected ({len(ptr) - 1},)"
                )

    @classmethod
    def from_arrays(
        cls,
        positions: ArrayLike,
        atomic_numbers: ArrayLike,
        ptr: ArrayLike,
        total_charge: ArrayLike | None = None,
        multiplicity: ArrayLike | None = None,
    ) -> PackedStructures:
        """Cast to the stored dtypes and validate the shapes."""
        return cls(
            positions=np.ascontiguousarray(positions, dtype=POSITION_DTYPE),
            atomic_numbers=np.ascontiguousarray(
                atomic_numbers, dtype=ATOMIC_NUMBER_DTYPE
            ),
            ptr=np.ascontiguousarray(ptr, dtype=np.int64),
            total_charge=_per_structure(total_charge),
            multiplicity=_per_structure(multiplicity),
        )

    @classmethod
    def from_frames(
        cls, frames: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> PackedStructures:
        """From ``(atomic_numbers, positions)`` pairs, one per structure."""
        counts = np.array([len(numbers) for numbers, _ in frames], dtype=np.int64)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        if not frames:
            return cls.empty()
        return cls.from_arrays(
            np.concatenate([np.asarray(positions) for _, positions in frames]),
            np.concatenate([np.asarray(numbers) for numbers, _ in frames]),
            ptr,
        )

    @classmethod
    def from_atoms(cls, structures: Sequence[Atoms]) -> PackedStructures:
        return cls.from_frames(
            [
                (atoms.get_atomic_numbers(), atoms.get_positions())
                for atoms in structures
            ]
        )

    @classmethod
    def empty(cls) -> PackedStructures:
        return cls.from_arrays(
            np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.int64)
        )

    @classmethod
    def concatenate(cls, parts: Sequence[PackedStructures]) -> PackedStructures:
        """The structures of ``parts`` in order, as one block.

        Raises:
            ValueError: if some parts carry charges and multiplicities and
                others do not.
        """
        if not parts:
            return cls.empty()
        offsets = np.cumsum([0] + [int(part.ptr[-1]) for part in parts[:-1]])
        ptr = np.concatenate(
            [[0]]
            + [
                part.ptr[1:] + offset
                for part, offset in zip(parts, offsets, strict=True)
            ]
        )
        per_structure: dict[str, np.ndarray | None] = {}
        for name in ("total_charge", "multiplicity"):
            values = [getattr(part, name) for part in parts]
            present = [value for value in values if value is not None]
            if present and len(present) != len(values):
                raise ValueError(f"only some parts carry {name}")
            per_structure[name] = np.concatenate(present) if present else None
        return cls.from_arrays(
            np.concatenate([part.positions for part in parts]),
            np.concatenate([part.atomic_numbers for part in parts]),
            ptr,
            **per_structure,
        )

    def with_charges(
        self, total_charge: ArrayLike, multiplicity: ArrayLike
    ) -> PackedStructures:
        """These structures with their total charges and multiplicities."""
        return PackedStructures.from_arrays(
            self.positions, self.atomic_numbers, self.ptr, total_charge, multiplicity
        )

    def __len__(self) -> int:
        return len(self.ptr) - 1

    @property
    def atom_counts(self) -> np.ndarray:
        return np.diff(self.ptr)

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """``(atomic_numbers, positions)`` of structure ``index`` (views)."""
        start, stop = int(self.ptr[index]), int(self.ptr[index + 1])
        return self.atomic_numbers[start:stop], self.positions[start:stop]

    def frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for index in range(len(self)):
            yield self.frame(index)

    def atoms(self, index: int) -> Atoms:
        numbers, positions = self.frame(index)
        return Atoms(numbers=numbers, positions=positions, pbc=[0, 0, 0])

    def slice(self, start: int, stop: int) -> PackedStructures:
        """Structures ``start:stop``, ``ptr`` rebased to 0."""
        return self.take(np.arange(start, stop))

    def take(self, indices: Sequence[int] | np.ndarray) -> PackedStructures:
        """The structures at ``indices``, in that order (copies)."""
        indices = np.asarray(indices, dtype=np.int64)
        ptr, atom_index = gather_index(self.ptr, indices)
        return PackedStructures(
            positions=self.positions[atom_index],
            atomic_numbers=self.atomic_numbers[atom_index],
            ptr=ptr,
            total_charge=None
            if self.total_charge is None
            else self.total_charge[indices],
            multiplicity=None
            if self.multiplicity is None
            else self.multiplicity[indices],
        )

    def save_npz(self, path) -> None:
        """The frames (not the charges, which the table holds) to ``path``."""
        np.savez(
            path,
            positions=self.positions,
            atomic_numbers=self.atomic_numbers,
            ptr=self.ptr,
        )

    @classmethod
    def load_npz(cls, path) -> PackedStructures:
        with np.load(path) as data:
            return cls.from_arrays(
                data["positions"], data["atomic_numbers"], data["ptr"]
            )
