"""PCQM4Mv2: the 3,378,606 OGB-LSC training molecules with DFT geometries.

The raw files are ``pcqm4m-v2-train.sdf`` (md5-checked against OGB) and the
OGB ``data.csv.gz`` (``idx,smiles,homolumogap``), whose first 3,378,606 rows
are the SDF records in order (``download_pcqm4mv2.sh``). Record ``i`` of the
SDF is ``source_key`` ``i``. Shards read byte ranges of the SDF through an
index of record offsets that the first run writes into the work directory.
"""

from __future__ import annotations

import mmap
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from rdkit import Chem

from remedi_data.bundle import LabelColumn, TaskType
from remedi_data.chemistry.elements import ElementSet
from remedi_data.chemistry.geometry import GeometryLimits
from remedi_data.chemistry.smiles_filter import SmilesFilterConfig
from remedi_data.structures.records import UNREADABLE, DroppedRecord, StructureRecord
from remedi_data.structures.source import StructureSourceBase

OFFSETS_FILENAME = "sdf_record_offsets.npy"
GAPS_FILENAME = "homo_lumo_gap.npy"
RECORD_END = b"\n$$$$"


def sdf_record_offsets(path: Path) -> np.ndarray:
    """Byte offset of every record start, plus the file size as the last entry."""
    offsets = [0]
    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        size = len(mapped)
        position = mapped.find(RECORD_END)
        while position != -1:
            line_end = mapped.find(b"\n", position + len(RECORD_END))
            next_start = size if line_end == -1 else line_end + 1
            if next_start < size:
                offsets.append(next_start)
            position = mapped.find(RECORD_END, next_start - 1)
        if offsets[-1] != size:
            offsets.append(size)
    return np.asarray(offsets, dtype=np.int64)


def _pcqm_smiles_filter() -> SmilesFilterConfig:
    # Neutralising would move hydrogens the geometry does not move; a charged
    # molecule keeps its charge instead (total_charge is stored per structure).
    # Radicals are dropped: ~7.7% of PCQM4Mv2 SMILES carry radical centres from
    # PCQM's own bond perception, while the DFT was closed-shell, so the
    # multiplicity derived from them (3-19) would be wrong (user, 2026-11-16).
    return SmilesFilterConfig(
        max_atoms=100,
        element_set=ElementSet.mace_off,
        allow_radicals=False,
        strip_salts=True,
        neutralize=False,
        dedupe=False,
    )


def _pcqm_geometry_limits() -> GeometryLimits:
    return GeometryLimits(elements=ElementSet.mace_off, min_interatomic_distance=0.5)


class Pcqm4mv2Source(StructureSourceBase):
    """PCQM4Mv2 training molecules from the SDF, with the HOMO-LUMO gap as label."""

    kind: Literal["pcqm4mv2"] = "pcqm4mv2"
    dataset_id: str = "pcqm4mv2"
    description: str = (
        "PCQM4Mv2 (OGB-LSC) training molecules with their DFT (B3LYP/6-31G*) "
        "geometries and HOMO-LUMO gaps"
    )
    sdf: Path
    labels_csv: Path
    smiles_filter: SmilesFilterConfig | None = Field(
        default_factory=_pcqm_smiles_filter
    )
    geometry_limits: GeometryLimits = Field(default_factory=_pcqm_geometry_limits)

    def labels(self) -> list[LabelColumn]:
        return [LabelColumn(name="homo_lumo_gap", task_type=TaskType.regression)]

    def raw_files(self) -> list[Path]:
        return [self.sdf, self.labels_csv]

    def geometry_description(self) -> str:
        return "PCQM4Mv2 training SDF: DFT (B3LYP/6-31G*) equilibrium geometries"

    def record_count(self, work_directory: Path) -> int:
        """Index the SDF and cache the gaps; both land in ``work_directory``."""
        work_directory.mkdir(parents=True, exist_ok=True)
        offsets = sdf_record_offsets(self.sdf)
        np.save(work_directory / OFFSETS_FILENAME, offsets)
        labels = pd.read_csv(self.labels_csv, usecols=["idx", "homolumogap"])
        if not np.array_equal(labels["idx"].to_numpy(), np.arange(len(labels))):
            raise ValueError(f"{self.labels_csv}: idx is not 0 .. N-1 in order")
        record_count = len(offsets) - 1
        if record_count > len(labels):
            raise ValueError(
                f"{self.sdf} holds {record_count} records but {self.labels_csv} "
                f"only {len(labels)} rows"
            )
        np.save(
            work_directory / GAPS_FILENAME,
            labels["homolumogap"].to_numpy(dtype=np.float64)[:record_count],
        )
        return record_count

    def read_records(
        self, start: int, stop: int, work_directory: Path
    ) -> Iterator[StructureRecord | DroppedRecord]:
        offsets = np.load(work_directory / OFFSETS_FILENAME)
        gaps = np.load(work_directory / GAPS_FILENAME)
        with self.sdf.open("rb") as handle:
            handle.seek(int(offsets[start]))
            text = handle.read(int(offsets[stop] - offsets[start])).decode()
        supplier = Chem.SDMolSupplier()
        supplier.SetData(text, sanitize=True, removeHs=False)
        count = 0
        for offset, molecule in enumerate(supplier):
            index = start + offset
            count += 1
            if molecule is None:
                yield DroppedRecord(source_key=str(index), reason=UNREADABLE)
                continue
            yield StructureRecord(
                source_key=str(index),
                molecule=molecule,
                labels={"homo_lumo_gap": float(gaps[index])},
            )
        if count != stop - start:
            raise ValueError(
                f"{self.sdf}: bytes of records {start}:{stop} parsed into {count} "
                "records; the offset index is stale"
            )
