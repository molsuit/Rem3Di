"""``PackedStructures``: many structures as three flat arrays, no per-structure objects.

The in-memory form the writer, the validator and the structure preparers pass
around. Structure ``i`` owns atoms ``ptr[i]:ptr[i + 1]`` of ``positions`` and
``atomic_numbers``. The dtypes are the ones the zarr stores (float32 positions,
uint8 atomic numbers), so whatever is checked in memory is exactly what is
written.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

POSITION_DTYPE = np.float32
ATOMIC_NUMBER_DTYPE = np.uint8


@dataclass(frozen=True)
class PackedStructures:
    """``positions`` ``(n_atoms, 3)`` float32, ``atomic_numbers`` ``(n_atoms,)`` uint8,
    ``ptr`` ``(n_structures + 1,)`` int64 starting at 0."""

    positions: np.ndarray
    atomic_numbers: np.ndarray
    ptr: np.ndarray

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

    @classmethod
    def from_arrays(
        cls, positions: np.ndarray, atomic_numbers: np.ndarray, ptr: np.ndarray
    ) -> PackedStructures:
        """Cast to the stored dtypes and validate the shapes."""
        return cls(
            positions=np.ascontiguousarray(positions, dtype=POSITION_DTYPE),
            atomic_numbers=np.ascontiguousarray(
                atomic_numbers, dtype=ATOMIC_NUMBER_DTYPE
            ),
            ptr=np.ascontiguousarray(ptr, dtype=np.int64),
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
        """The structures of ``parts`` in order, as one block."""
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
        return cls.from_arrays(
            np.concatenate([part.positions for part in parts]),
            np.concatenate([part.atomic_numbers for part in parts]),
            ptr,
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
        """Structures ``start:stop`` (copies, ``ptr`` rebased to 0)."""
        atom_start, atom_stop = int(self.ptr[start]), int(self.ptr[stop])
        return PackedStructures(
            positions=self.positions[atom_start:atom_stop].copy(),
            atomic_numbers=self.atomic_numbers[atom_start:atom_stop].copy(),
            ptr=self.ptr[start : stop + 1] - atom_start,
        )

    def take(self, indices: Sequence[int] | np.ndarray) -> PackedStructures:
        """The structures at ``indices``, in that order."""
        indices = np.asarray(indices, dtype=np.int64)
        counts = self.atom_counts[indices]
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        atom_index = (
            np.repeat(self.ptr[indices] - ptr[:-1], counts) + np.arange(ptr[-1])
            if len(indices)
            else np.zeros(0, dtype=np.int64)
        )
        return PackedStructures(
            positions=self.positions[atom_index],
            atomic_numbers=self.atomic_numbers[atom_index],
            ptr=ptr,
        )

    def save_npz(self, path) -> None:
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
