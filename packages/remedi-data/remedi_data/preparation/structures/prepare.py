"""One yaml prepares several structure corpora: one task per source, through ``run_tasks``.

A failing source is one failed row in ``status.yaml`` under ``work_root``;
the others still finish. Each source's shards live in
``work_root/<dataset_id>/`` and its dataset in ``output_root/<dataset_id>/``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from remedi_data.preparation.structures.pipeline import (
    StructurePreparationSettings,
    prepare_structures,
)
from remedi_data.preparation.structures.sources import StructureSource
from remedi_data.runs.results import EvalResult, PydanticResult
from remedi_data.runs.task_runner import (
    RunReport,
    run_and_record,
)

logger = logging.getLogger(__name__)


class StructurePreparationConfig(BaseModel):
    """The yaml of one preparation run."""

    model_config = ConfigDict(extra="forbid")

    output_root: Path
    #: Shards, indexes and the run's ``status.yaml``; safe to delete after a run.
    work_root: Path
    sources: list[StructureSource] = Field(min_length=1)
    settings: StructurePreparationSettings = Field(
        default_factory=StructurePreparationSettings
    )
    overwrite: bool = False
    keep_going: bool = True


class PrepareStructuresTask(BaseModel):
    """Prepare one source; the summary is the task's artifact."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["prepare_structures"] = "prepare_structures"
    source: StructureSource
    config: StructurePreparationConfig

    @property
    def status_label(self) -> str:
        return self.source.dataset_id

    def run(self, ctx: object) -> Iterator[EvalResult]:
        summary = prepare_structures(
            self.source,
            self.config.settings,
            self.config.output_root / self.source.dataset_id,
            self.config.work_root / self.source.dataset_id,
            overwrite=self.config.overwrite,
        )
        yield PydanticResult(
            file_name=Path("prepared") / f"{self.status_label}.yaml", obj=summary
        )


def prepare_structure_datasets(
    config: StructurePreparationConfig, only: list[str] | None = None
) -> RunReport:
    """Prepare every source (or those named in ``only``); writes ``status.yaml``."""
    sources = [
        source for source in config.sources if not only or source.dataset_id in only
    ]
    if only and len(sources) != len(set(only)):
        known = [source.dataset_id for source in config.sources]
        raise ValueError(f"--only {only} names sources not in the config {known}")
    tasks = [PrepareStructuresTask(source=source, config=config) for source in sources]
    return run_and_record(
        [tasks], None, config.work_root, config, keep_going=config.keep_going
    )
