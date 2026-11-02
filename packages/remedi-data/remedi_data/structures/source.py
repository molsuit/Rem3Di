"""The base every structure source config derives from.

A source is a pydantic model (so a preparation config stays declarative yaml)
that also knows how to read its raw files: how many records they hold and the
records ``start:stop``. The pipeline shards on that numbering, so a source
needs a stable, cheap record order. The source object is pickled into the
worker processes, so it must stay a plain config; anything expensive to build
(an index of the SDF records) goes into ``work_directory``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from remedi_data.bundle.spec import LabelColumn
from remedi_data.chemistry.geometry import GeometryLimits
from remedi_data.chemistry.smiles_filter import SmilesFilterConfig
from remedi_data.structures.records import DroppedRecord, StructureRecord


class StructureSourceBase(BaseModel):
    """Fields every source shares; subclasses add a ``kind`` literal and their raw paths.

    ``smiles_filter`` set means the dataset carries SMILES derived from each
    structure (``smiles: true``, ids from the SMILES); ``None`` means it does
    not, and ids come from ``source_key``.
    """

    model_config = ConfigDict(extra="forbid")

    #: Narrowed to a ``Literal`` discriminator by every source.
    kind: str
    dataset_id: str = Field(min_length=1)
    description: str = ""
    smiles_filter: SmilesFilterConfig | None = None
    geometry_limits: GeometryLimits = Field(default_factory=GeometryLimits)
    #: Read only the first ``limit`` records; for trial runs and the old-path checks.
    limit: int | None = Field(default=None, ge=1)

    def labels(self) -> list[LabelColumn]:
        """The label columns, in table order."""
        return []

    def extra_columns(self) -> dict[str, str]:
        """Extra columns besides ``source_key``: name -> pandas dtype, in table order."""
        return {}

    def raw_files(self) -> list[Path]:
        """The files whose sha256 goes into ``provenance.yaml``."""
        raise NotImplementedError

    def geometry_description(self) -> str:
        """Free text for ``provenance.source.geometry``."""
        raise NotImplementedError

    def record_count(self, work_directory: Path) -> int:
        """How many records the raw files hold (before ``limit``)."""
        raise NotImplementedError

    def read_records(
        self, start: int, stop: int, work_directory: Path
    ) -> Iterator[StructureRecord | DroppedRecord]:
        """The records ``start:stop``. A record may yield several structures
        (conformers); every structure is counted as read."""
        raise NotImplementedError
