"""Preparers for sources that ship structures (``BENCHMARK_DATA_FORMAT.md`` §5.2.2).

A source (:mod:`remedi_data.structures.sources`) reads its raw files into
records; the pipeline (:mod:`remedi_data.structures.pipeline`) derives SMILES
from each geometry, gates it, writes restartable shards in a process pool and
assembles the dataset through ``write_dataset``; :mod:`remedi_data.structures.prepare`
runs one yaml of sources as tasks.
"""

from remedi_data.structures.pipeline import (
    PreparationSummary,
    ShardFailures,
    StructurePreparationSettings,
    prepare_structures,
)
from remedi_data.structures.prepare import (
    PrepareStructuresTask,
    StructurePreparationConfig,
    prepare_structure_datasets,
)
from remedi_data.structures.records import (
    DroppedRecord,
    PreparedFrame,
    StructureRecord,
    prepare_record,
)
from remedi_data.structures.source import StructureSourceBase
from remedi_data.structures.sources import (
    GeomDrugsSource,
    Pcqm4mv2Source,
    StructureSource,
    TmqmSource,
)

__all__ = [
    "DroppedRecord",
    "GeomDrugsSource",
    "Pcqm4mv2Source",
    "PreparationSummary",
    "PrepareStructuresTask",
    "PreparedFrame",
    "ShardFailures",
    "StructurePreparationConfig",
    "StructurePreparationSettings",
    "StructureRecord",
    "StructureSource",
    "StructureSourceBase",
    "TmqmSource",
    "prepare_record",
    "prepare_structure_datasets",
    "prepare_structures",
]
