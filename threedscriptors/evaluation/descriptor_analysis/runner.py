from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis.analysis_tasks import (
    DescriptorAnalysisTask,
)
from threedscriptors.evaluation.descriptor_analysis.context import (
    DescriptorAnalysisContext,
    DescriptorNormalizationConfig,
    ProjectionConfig,
)
from threedscriptors.evaluation.results import EvalResult


class DescriptorAnalysisRunner(BaseModel):
    """Run a list of descriptor-analysis tasks against precomputed descriptors.

    The runner owns the (single) descriptor normalization + 2D projection and
    hands a shared :class:`DescriptorAnalysisContext` to each task. Tasks return
    :class:`EvalResult` instances that the runner collects and serializes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tasks: list[DescriptorAnalysisTask]
    normalization: DescriptorNormalizationConfig = DescriptorNormalizationConfig()
    projection: ProjectionConfig | None = ProjectionConfig()
    file_prefix: str = ""

    def run(
        self,
        descriptors: np.ndarray | torch.Tensor,
        dataset: MoleculeDataset,
    ) -> list[EvalResult]:
        ctx = DescriptorAnalysisContext.build(
            dataset=dataset,
            descriptors=descriptors,
            normalization=self.normalization,
            projection_config=self.projection,
            file_prefix=self.file_prefix,
        )

        results: list[EvalResult] = []
        for task in self.tasks:
            task_label = type(task).__name__
            try:
                results.extend(task.run(ctx))
            except Exception:
                logger.error(
                    "Descriptor analysis task %s failed; continuing.\n%s",
                    task_label,
                    traceback.format_exc(),
                )
        return results

    @staticmethod
    def serialize(results: list[EvalResult], output_dir: str | Path) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            result.serialize_to(out_dir)
