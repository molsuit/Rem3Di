"""tmQM: transition-metal complexes from the CSD with DFT geometries and properties.

The raw files are the pinned ``tmQM_X{1,2,3}.xyz.gz`` and ``tmQM_y.csv`` of
github.com/uiocompcat/tmQM (``download_tmqm.sh``). Each xyz entry's comment line
reads ``CSD_code = WELROW | q = 0 | S = 0 | Stoichiometry = ... | MND = 8 | ...``;
entries are separated by blank lines. The dataset declares ``smiles: false``
(the CSD SMILES with dative bonds is kept as the extra column ``tmqm_smiles``),
so the ids come from the CSD code.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from pydantic import Field

from remedi_data.bundle.spec import LabelColumn
from remedi_data.chemistry.geometry import GeometryLimits
from remedi_data.dataset.tasks import TaskType
from remedi_data.structures.records import DroppedRecord, StructureRecord
from remedi_data.structures.source import StructureSourceBase

#: Columns of ``tmQM_y.csv`` that become labels, all regression, in table order.
TMQM_LABELS: tuple[str, ...] = (
    "Electronic_E",
    "Dispersion_E",
    "Dipole_M",
    "Metal_q",
    "HL_Gap",
    "HOMO_Energy",
    "LUMO_Energy",
    "Polarizability",
)
XYZ_FILES: tuple[str, ...] = ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz")
LABELS_FILE = "tmQM_y.csv"
#: ``counts.dropped`` key for an entry whose xyz block cannot be parsed.
MALFORMED_XYZ_ENTRY = "malformed_xyz_entry"


def _tmqm_geometry_limits() -> GeometryLimits:
    # The old tmQM build's limits (load_from_xyz_tmqm.py), plus the overlap
    # guard; any element, since tmQM is a curated CSD source.
    return GeometryLimits(
        max_atoms=200,
        reject_zero_hydrogen=True,
        min_hydrogen_heavy_ratio=0.1,
        min_interatomic_distance=0.5,
    )


def parse_comment(comment: str) -> dict[str, str]:
    """``key = value | key = value`` -> dict; fields without ``=`` are skipped."""
    fields: dict[str, str] = {}
    for part in comment.split("|"):
        key, separator, value = part.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def iterate_xyz_entries(lines: Iterator[str]) -> Iterator[tuple[str, list[str]]]:
    """``(comment, atom lines)`` per entry; blank lines between entries are skipped.

    Raises:
        ValueError: if an entry's atom-count line is not an integer or the file
            ends inside an entry.
    """
    for line in lines:
        if not line.strip():
            continue
        count = int(line.strip())
        comment = next(lines).rstrip("\n")
        atom_lines = [next(lines) for _ in range(count)]
        yield comment, atom_lines


class TmqmSource(StructureSourceBase):
    """tmQM from its pinned raw directory."""

    kind: Literal["tmqm"] = "tmqm"
    dataset_id: str = "tmqm"
    description: str = (
        "tmQM: transition-metal complexes from the CSD, DFT-optimised geometries "
        "(GFN2-xTB, then single points), eight properties"
    )
    directory: Path
    geometry_limits: GeometryLimits = Field(default_factory=_tmqm_geometry_limits)

    def labels(self) -> list[LabelColumn]:
        return [
            LabelColumn(name=name, task_type=TaskType.regression)
            for name in TMQM_LABELS
        ]

    def extra_columns(self) -> dict[str, str]:
        return {
            "csd_years": "str",
            "tmqm_smiles": "str",
            "metal_neighbour_count": "Int64",
        }

    def raw_files(self) -> list[Path]:
        return [self.directory / name for name in (*XYZ_FILES, LABELS_FILE)]

    def geometry_description(self) -> str:
        return "tmQM xyz files: GFN2-xTB optimised geometries"

    def _entries(self) -> Iterator[tuple[str, list[str]]]:
        for name in XYZ_FILES:
            with gzip.open(self.directory / name, "rt") as handle:
                yield from iterate_xyz_entries(iter(handle))

    def record_count(self, work_directory: Path) -> int:
        return sum(1 for _ in self._entries())

    def _labels_by_code(self) -> pd.DataFrame:
        table = pd.read_csv(self.directory / LABELS_FILE, sep=";")
        return table.drop_duplicates("CSD_code").set_index("CSD_code")

    def read_records(
        self, start: int, stop: int, work_directory: Path
    ) -> Iterator[StructureRecord | DroppedRecord]:
        labels = self._labels_by_code()
        for position, (comment, atom_lines) in enumerate(self._entries()):
            if position < start:
                continue
            if position >= stop:
                return
            fields = parse_comment(comment)
            code = fields.get("CSD_code", f"entry_{position}")
            try:
                symbols = [line.split()[0] for line in atom_lines]
                positions = np.array(
                    [
                        [float(value) for value in line.split()[1:4]]
                        for line in atom_lines
                    ]
                )
                numbers = np.array([atomic_numbers[symbol] for symbol in symbols])
                charge = float(fields["q"])
                spin = float(fields["S"])
            except (KeyError, ValueError, IndexError):
                yield DroppedRecord(source_key=code, reason=MALFORMED_XYZ_ENTRY)
                continue
            row = labels.loc[code] if code in labels.index else None
            yield StructureRecord(
                source_key=code,
                atomic_numbers=numbers,
                positions=positions,
                total_charge=charge,
                multiplicity=2.0 * spin + 1.0,
                labels=(
                    {}
                    if row is None
                    else {name: float(row[name]) for name in TMQM_LABELS}
                ),
                extras={
                    "csd_years": None
                    if row is None
                    else _text_or_none(row["CSD_years"]),
                    "tmqm_smiles": None
                    if row is None
                    else _text_or_none(row["SMILES"]),
                    "metal_neighbour_count": _integer_or_none(fields.get("MND")),
                },
            )


def _text_or_none(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def _integer_or_none(value: str | None) -> int | None:
    try:
        return None if value is None else int(value)
    except ValueError:
        return None
