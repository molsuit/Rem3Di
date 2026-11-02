"""GEOM-Drugs: CREST conformer ensembles of drug-like molecules.

The raw directory is the extracted ``rdkit_folder`` of the GEOM dataverse
release (``download_geomdrugs.sh``): ``summary_drugs.json`` maps each SMILES to
its ``pickle_path``, and each pickle holds ``conformers``, a list of dicts with
an ``rd_mol`` (explicit hydrogens, one conformer) and its ``boltzmannweight``.
Record ``i`` is the ``i``-th molecule of the summary; each yields its
``conformers_per_molecule`` most Boltzmann-weighted conformers.

The pickles hold RDKit molecules, so reading them unpickles the downloaded
files: only point this at the pinned GEOM release.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field

from remedi_data.chemistry.elements import ElementSet
from remedi_data.chemistry.geometry import GeometryLimits
from remedi_data.chemistry.smiles_filter import SmilesFilterConfig
from remedi_data.structures.records import DroppedRecord, StructureRecord
from remedi_data.structures.source import StructureSourceBase

SUMMARY_FILENAME = "summary_drugs.json"
INDEX_FILENAME = "geom_molecules.parquet"
#: ``counts.dropped`` keys of molecules whose pickle is missing or unreadable.
MISSING_PICKLE = "missing_pickle"
UNREADABLE_PICKLE = "unreadable_pickle"


def _geom_smiles_filter() -> SmilesFilterConfig:
    return SmilesFilterConfig(
        max_atoms=None,
        element_set=ElementSet.mace_off,
        allow_charged=False,
        strip_salts=True,
        neutralize=False,
        dedupe=False,
    )


def _geom_geometry_limits() -> GeometryLimits:
    return GeometryLimits(
        max_atoms=128, elements=ElementSet.mace_off, min_interatomic_distance=0.5
    )


class GeomDrugsSource(StructureSourceBase):
    """GEOM-Drugs, the top conformers of each molecule by Boltzmann weight."""

    kind: Literal["geom_drugs"] = "geom_drugs"
    dataset_id: str = "geom_drugs"
    description: str = (
        "GEOM-Drugs: CREST/GFN2-xTB conformers of drug-like molecules, the most "
        "Boltzmann-weighted conformers per molecule"
    )
    #: The extracted ``rdkit_folder``.
    directory: Path
    conformers_per_molecule: int = Field(default=5, ge=1)
    smiles_filter: SmilesFilterConfig | None = Field(
        default_factory=_geom_smiles_filter
    )
    geometry_limits: GeometryLimits = Field(default_factory=_geom_geometry_limits)

    def extra_columns(self) -> dict[str, str]:
        return {
            "geom_smiles": "str",
            "conformer_rank": "Int64",
            "boltzmann_weight": "Float64",
        }

    def raw_files(self) -> list[Path]:
        return [self.directory / SUMMARY_FILENAME]

    def geometry_description(self) -> str:
        return "GEOM-Drugs rdkit_folder pickles: CREST/GFN2-xTB conformers"

    def record_count(self, work_directory: Path) -> int:
        """Write the molecule index (summary order) into ``work_directory``."""
        with (self.directory / SUMMARY_FILENAME).open() as handle:
            summary = json.load(handle)
        index = pd.DataFrame(
            {
                "geom_smiles": list(summary),
                "pickle_path": [entry.get("pickle_path") for entry in summary.values()],
            }
        )
        work_directory.mkdir(parents=True, exist_ok=True)
        index.to_parquet(work_directory / INDEX_FILENAME, index=False)
        return len(index)

    def read_records(
        self, start: int, stop: int, work_directory: Path
    ) -> Iterator[StructureRecord | DroppedRecord]:
        index = pd.read_parquet(work_directory / INDEX_FILENAME).iloc[start:stop]
        for geom_smiles, pickle_path in zip(
            index["geom_smiles"], index["pickle_path"], strict=True
        ):
            path = None if pickle_path is None else self.directory / pickle_path
            if path is None or not path.exists():
                yield DroppedRecord(source_key=geom_smiles, reason=MISSING_PICKLE)
                continue
            try:
                with path.open("rb") as handle:
                    conformers = pickle.load(handle)["conformers"]
            except Exception:  # a corrupt pickle is one dropped molecule
                yield DroppedRecord(source_key=geom_smiles, reason=UNREADABLE_PICKLE)
                continue
            ranked = sorted(
                conformers, key=lambda conformer: -conformer["boltzmannweight"]
            )
            for rank, conformer in enumerate(ranked[: self.conformers_per_molecule]):
                yield StructureRecord(
                    source_key=f"{geom_smiles}#{rank}",
                    molecule=conformer["rd_mol"],
                    extras={
                        "geom_smiles": geom_smiles,
                        "conformer_rank": rank,
                        "boltzmann_weight": float(conformer["boltzmannweight"]),
                    },
                )
