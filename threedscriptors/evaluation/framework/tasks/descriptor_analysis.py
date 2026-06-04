"""Descriptor-analysis task: capacity / distribution / projection over one model.

Wraps the existing ``descriptor_analysis`` task family into the framework. It
builds the shared embedding once (via :class:`EmbeddingSpec`), constructs a
:class:`DescriptorAnalysisContext` from it, and runs the configured analysis
sub-tasks — each in its own try/except so one failing analysis does not lose the
others. Sub-task artifacts are re-rooted under ``descriptor_analysis/``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis import (
    DescriptorAnalysisContext,
    DescriptorAnalysisTask,
    DescriptorNormalizationConfig,
    ProjectionConfig,
)
from threedscriptors.evaluation.framework.context import EvalContext
from threedscriptors.evaluation.framework.resources import EmbeddingSpec
from threedscriptors.evaluation.results import EvalResult

logger = logging.getLogger(__name__)


class DescriptorAnalysisConfig(BaseModel):
    """Intrinsic descriptor analyses (capacity, distribution, projection, ...)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["descriptor_analysis"] = "descriptor_analysis"
    dataset_path: Path
    dataset_id: str
    normalization: DescriptorNormalizationConfig = Field(
        default_factory=DescriptorNormalizationConfig
    )
    # 2D projection for projection/chemiscope tasks. None (default) skips the
    # UMAP fit — fine for capacity / distribution tasks that don't need it.
    projection: ProjectionConfig | None = None
    max_structures: int | None = None
    tasks: list[DescriptorAnalysisTask] = Field(default_factory=list, min_length=1)

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        ctx.task_dir("descriptor_analysis")
        dataset = MoleculeDataset.open_existing_dataset_from_dir(self.dataset_path)
        X = ctx.resources.get(
            EmbeddingSpec(
                dataset_id=self.dataset_id,
                descriptor=ctx.model,
                dataset=dataset,
                cache_dir=ctx.resource_cache_dir,
            )
        )
        sample_indices = None
        if self.max_structures is not None and self.max_structures < X.shape[0]:
            X = X[: self.max_structures]
            sample_indices = np.arange(X.shape[0], dtype=np.int64)

        da_ctx = DescriptorAnalysisContext.build(
            dataset=dataset,
            descriptors=X,
            normalization=self.normalization,
            projection_config=self.projection,
            sample_indices=sample_indices,
        )

        for task in self.tasks:
            label = type(task).__name__
            try:
                results = task.run(da_ctx)
            except Exception:  # one analysis must not lose the others
                logger.exception("descriptor-analysis %s failed; skipping", label)
                continue
            for r in results:
                yield r.model_copy(
                    update={"file_name": Path("descriptor_analysis") / r.file_name}
                )
