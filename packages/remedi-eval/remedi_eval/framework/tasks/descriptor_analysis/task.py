"""``DescriptorAnalysisConfig``: the descriptor-space analyses as one eval task.

It reads the run's ``model`` descriptor over one dataset through
:meth:`EvalContext.descriptor_matrix`, so any descriptor the eval framework knows
works: an ECFP fingerprint, precomputed embeddings, or a Rem3Di checkpoint's
embeddings (written by ``remedi.embedding.embed_and_evaluate``). It builds one :class:`DescriptorAnalysisContext` (normalisation, optional 2D
projection) and runs the configured analyses. A failing analysis does not lose
the others: they all run and write their artifacts, then the task raises
:class:`DescriptorAnalysisError` naming every failure, so the run records it.
Artifacts go under ``descriptor_analysis/``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from remedi_data.bundle import read_spec
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.runs.results import EvalResult
from remedi_eval.framework.context import EvalContext
from remedi_eval.framework.tasks.descriptor_analysis.analysis_tasks import (
    DescriptorAnalysisTask,
)
from remedi_eval.framework.tasks.descriptor_analysis.context import (
    DescriptorAnalysisContext,
    DescriptorNormalizationConfig,
)
from remedi_eval.projection import ProjectionConfig

logger = logging.getLogger(__name__)


class DescriptorAnalysisError(RuntimeError):
    """One or more analyses of a :class:`DescriptorAnalysisConfig` failed."""


class DescriptorAnalysisConfig(BaseModel):
    """Intrinsic descriptor analyses (capacity, distribution, projection, ...)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["descriptor_analysis"] = "descriptor_analysis"
    dataset_path: Path
    normalization: DescriptorNormalizationConfig = Field(
        default_factory=DescriptorNormalizationConfig
    )
    # 2D projection for projection/chemiscope tasks. None (default) skips the
    # UMAP fit — fine for capacity / distribution tasks that don't need it.
    projection: ProjectionConfig | None = None
    max_structures: int | None = Field(default=None, gt=0)
    """Analyse only the first ``max_structures`` rows (the whole dataset is embedded)."""
    tasks: list[DescriptorAnalysisTask] = Field(default_factory=list, min_length=1)

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        ctx.task_dir("descriptor_analysis")
        dataset = MoleculeDataset.open_existing_dataset_from_dir(self.dataset_path)
        descriptors = ctx.descriptor_matrix(
            ctx.model, dataset, read_spec(self.dataset_path).dataset_id
        )
        if self.max_structures is not None:
            descriptors = descriptors[: self.max_structures]

        analysis_context = DescriptorAnalysisContext.build(
            dataset=dataset,
            descriptors=descriptors,
            normalization=self.normalization,
            projection_config=self.projection,
        )

        failures: list[str] = []
        for task in self.tasks:
            label = type(task).__name__
            try:
                results = task.run(analysis_context)
            except Exception as error:  # run the others, raise below
                logger.exception("descriptor analysis %s failed", label)
                failures.append(f"{label}: {error!r}")
                continue
            for result in results:
                yield result.model_copy(
                    update={"file_name": Path("descriptor_analysis") / result.file_name}
                )
        if failures:
            raise DescriptorAnalysisError(
                f"{len(failures)} of {len(self.tasks)} descriptor analyses failed: "
                + "; ".join(failures)
            )
