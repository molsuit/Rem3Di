"""``Dataset``: a handle on one dataset directory, and the structures' hash.

A dataset directory holds ``dataset.yaml``, ``table.parquet`` (one row per
structure, the only home of ids, charges and multiplicities),
``provenance.yaml`` and ``structures.atp`` (record ``i`` = the frame of table
row ``i``). :class:`Dataset` reads the frames through
:class:`~remedi_data.dataset.structure_file.StructureFile` and joins the charges
from the table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from remedi_data.dataset.structure_file import READ_CHUNK_STRUCTURES, StructureFile
from remedi_data.dataset.structures import Structures
from remedi_data.format.spec import CHARGE_COLUMNS, DatasetSpec
from remedi_data.format.table import (
    STRUCTURES_FILENAME,
    BundleValidationError,
    read_spec,
    read_table_polars,
    table_row_count,
)


def structures_sha256(
    frames: Structures | StructureFile,
    total_charge: ArrayLike,
    multiplicity: ArrayLike,
) -> str:
    """sha256 of everything the model reads, in record order (§3.3).

    The bytes of, in this order: the pointer (int64, little-endian), every
    ``atomic_numbers`` (uint8), every ``positions`` (float32, little-endian, C
    order), then the table's ``total_charge`` and ``multiplicity`` columns
    (each whole column as float32, little-endian). It moves whenever a frame,
    a charge or spin, the set of structures or their order changes. The frames
    are read in pieces, so memory stays bounded for any corpus.

    Raises:
        ValueError: if the charges do not have one entry per structure.
    """
    count = len(frames)
    charges = [
        np.ascontiguousarray(values, dtype="<f4")
        for values in (total_charge, multiplicity)
    ]
    for values in charges:
        if values.shape != (count,):
            raise ValueError(
                f"{values.shape} charges or multiplicities for {count} structures"
            )

    def pieces():
        for start in range(0, count, READ_CHUNK_STRUCTURES):
            yield frames.slice(start, start + READ_CHUNK_STRUCTURES)

    digest = hashlib.sha256()
    digest.update(np.zeros(1, dtype="<i8").tobytes())
    atoms_before = 0
    for piece in pieces():
        digest.update(np.ascontiguousarray(piece.ptr[1:] + atoms_before, "<i8"))
        atoms_before += int(piece.ptr[-1])
    for piece in pieces():
        digest.update(np.ascontiguousarray(piece.atomic_numbers, "u1"))
    for piece in pieces():
        digest.update(np.ascontiguousarray(piece.positions, "<f4"))
    for values in charges:
        digest.update(values)
    return digest.hexdigest()


@dataclass(frozen=True)
class Dataset:
    """One dataset directory: its spec, its frames and its table."""

    path: Path
    spec: DatasetSpec
    structures: StructureFile

    @classmethod
    def open(cls, path: Path) -> Dataset:
        """Open the dataset at ``path`` (read-only).

        Raises:
            FileNotFoundError: if ``dataset.yaml``, ``table.parquet`` or
                ``structures.atp`` is missing.
            BundleValidationError: if the spec cannot be read, or the
                structure file and the table disagree on the number of
                structures.
        """
        path = Path(path)
        spec = read_spec(path)
        structures = StructureFile.open(path / STRUCTURES_FILENAME)
        rows = table_row_count(path)
        if rows != len(structures):
            raise BundleValidationError(
                path,
                [
                    f"8: {STRUCTURES_FILENAME} holds {len(structures)} structures "
                    f"for {rows} table rows"
                ],
            )
        return cls(path=path, spec=spec, structures=structures)

    def __len__(self) -> int:
        return len(self.structures)

    def table(self, columns: list[str] | None = None) -> pl.DataFrame:
        """``table.parquet``, all of it or only ``columns``, row ``i`` = structure ``i``.

        Raises:
            polars.exceptions.ColumnNotFoundError: if a column is not in the table.
        """
        return read_table_polars(self.path, columns=columns)

    def read_structures(self, structure_ids: np.ndarray | None = None) -> Structures:
        """The frames of ``structure_ids`` (all by default) with their charges and
        multiplicities from the table, renumbered ``0..len - 1`` in that order.

        Raises:
            ValueError: if ``structure_ids`` is not sorted, unique and in range.
        """
        frames = self.structures.read(structure_ids)
        charges = self.table(list(CHARGE_COLUMNS))
        total_charge = charges["total_charge"].to_numpy()
        multiplicity = charges["multiplicity"].to_numpy()
        if structure_ids is not None:
            rows = np.asarray(structure_ids, dtype=np.int64)
            total_charge, multiplicity = total_charge[rows], multiplicity[rows]
        return frames.with_charges(total_charge, multiplicity)

    def structures_sha256(self) -> str:
        """:func:`structures_sha256` of this dataset's frames and table charges."""
        charges = self.table(list(CHARGE_COLUMNS))
        return structures_sha256(
            self.structures,
            charges["total_charge"].to_numpy(),
            charges["multiplicity"].to_numpy(),
        )
