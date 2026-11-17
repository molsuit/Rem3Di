"""``structures.atp``: the frames of a dataset, one atompack record per structure.

The only module that imports atompack (``BENCHMARK_DATA_FORMAT.md`` §3.3).
Record ``i`` is the structure of table row ``i``: its positions (float32,
``n_atoms x 3``) and atomic numbers (uint8), nothing else; ids, charges and
multiplicities live in ``table.parquet``. Swapping the storage (for example for
parquet list columns) changes only this module.

How the atompack 0.4.1 issues are avoided:

* #45, every ``flush()`` leaves a stale copy of the index in the file:
  :func:`write_structures` flushes exactly once and never appends to a finished
  file (a dataset is replaced whole).
* A write that dies before its flush leaves a file that opens as empty: the
  file is written as ``structures.atp.partial`` and renamed only after the
  flush and an fsync, so ``structures.atp`` either holds every record or does
  not exist.
* #48, a reader must not open a file another process is writing: a reader only
  ever sees the renamed, finished file.
* #54 and #55 (records stored out of order, flat reads of mixed optional
  fields): every record is a new molecule with the same two fields.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import atompack
import numpy as np

from remedi_data.dataset.structures import Structures

#: Structures per ``get_molecules_flat`` call when a whole file is read in
#: pieces (atom counts), so memory stays bounded for any corpus.
READ_CHUNK_STRUCTURES = 500_000


def partial_path(path: Path) -> Path:
    """Where :func:`write_structures` writes before the final rename."""
    return Path(path).with_name(f"{Path(path).name}.partial")


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_structures(path: Path, parts: Iterable[Structures]) -> int:
    """Write the frames of ``parts`` (in order) as a new file at ``path``.

    Streams: each part (one shard or chunk) becomes molecules that are added
    and then dropped, so memory holds one part plus atompack's index (20 bytes
    per structure). The index is flushed once, at the end; the file is fsynced
    and only then renamed from ``<path>.partial`` to ``path``. If anything
    fails, or ``parts`` raises, the partial file is removed and nothing is left
    at ``path``. Charges and multiplicities are not stored (the table holds
    them). Returns the number of structures written.

    Raises:
        FileExistsError: if ``path`` already exists; a finished file is never
            appended to or overwritten.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a structure file is written once")
    partial = partial_path(path)
    partial.unlink(missing_ok=True)  # left by an earlier write that died
    written = 0
    try:
        database = atompack.Database(str(partial), overwrite=True)
        for part in parts:
            molecules = [
                atompack.Molecule.from_arrays(positions, atomic_numbers)
                for atomic_numbers, positions in part.frames()
            ]
            database.add_molecules(molecules)
            written += len(part)
            del molecules, part
        database.flush()
        del database
        _fsync(partial)
        partial.rename(path)
        _fsync(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return written


def _pointer(atom_counts: np.ndarray) -> np.ndarray:
    pointer = np.zeros(len(atom_counts) + 1, dtype=np.int64)
    np.cumsum(atom_counts, out=pointer[1:])
    return pointer


class StructureFile:
    """A finished ``structures.atp``, read-only and memory-mapped.

    Pickles as its path and reopens on unpickling. A process that forks keeps
    the parent's mapping; code that reads from several processes (the
    data-loader workers) opens one per process instead
    (``TrainingMoleculeDataset`` does).
    """

    def __init__(self, path: Path, database: atompack.Database) -> None:
        self.path = Path(path)
        self._database = database

    @classmethod
    def open(cls, path: Path) -> StructureFile:
        """Open ``path`` read-only.

        Raises:
            FileNotFoundError: if there is no file at ``path``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        return cls(path, atompack.Database.open(str(path), mmap=True))

    def __reduce__(self):
        return (StructureFile.open, (self.path,))

    def __len__(self) -> int:
        return len(self._database)

    def _checked_ids(self, structure_ids: np.ndarray) -> np.ndarray:
        structure_ids = np.asarray(structure_ids).astype(np.int64, copy=False)
        if structure_ids.ndim != 1 or np.any(np.diff(structure_ids) <= 0):
            raise ValueError("structure_ids must be a sorted 1-d array without repeats")
        if len(structure_ids) and (
            structure_ids[0] < 0 or structure_ids[-1] >= len(self)
        ):
            raise ValueError(
                f"structure_ids reach outside the {len(self)} structures of {self.path}"
            )
        return structure_ids

    def read(self, structure_ids: np.ndarray | None = None) -> Structures:
        """The frames of ``structure_ids`` (all by default) in memory, renumbered
        ``0..len(structure_ids) - 1`` in that order; no charges.

        Raises:
            ValueError: if ``structure_ids`` is not sorted, unique and in range.
        """
        structure_ids = (
            np.arange(len(self), dtype=np.int64)
            if structure_ids is None
            else self._checked_ids(structure_ids)
        )
        # A list, not the array: atompack converts it about three times faster.
        flat = self._database.get_molecules_flat(structure_ids.tolist())
        return Structures.from_arrays(
            flat["positions"],
            flat["atomic_numbers"],
            _pointer(np.asarray(flat["n_atoms"], dtype=np.int64)),
        )

    def slice(self, start: int, stop: int) -> Structures:
        """The frames of structures ``start:stop``; ``stop`` past the end is
        clamped, as in a Python slice."""
        return self.read(np.arange(start, min(stop, len(self)), dtype=np.int64))

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """``(atomic_numbers, positions)`` of structure ``index``: uint8 and float32."""
        molecule = self._database[index]
        return molecule.atomic_numbers, molecule.positions

    def atom_counts(self, structure_ids: np.ndarray | None = None) -> np.ndarray:
        """Atoms per structure (int64), of ``structure_ids`` or of every structure.

        Read in pieces of :data:`READ_CHUNK_STRUCTURES`, so memory stays
        bounded.

        Raises:
            ValueError: if ``structure_ids`` is not sorted, unique and in range.
        """
        structure_ids = (
            np.arange(len(self), dtype=np.int64)
            if structure_ids is None
            else self._checked_ids(structure_ids)
        )
        counts = [
            np.asarray(
                self._database.get_molecules_flat(
                    structure_ids[start : start + READ_CHUNK_STRUCTURES].tolist()
                )["n_atoms"],
                dtype=np.int64,
            )
            for start in range(0, len(structure_ids), READ_CHUNK_STRUCTURES)
        ]
        return np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64)
