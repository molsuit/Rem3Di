from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

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
            results.extend(task.run(ctx))
        return results

    @staticmethod
    def serialize(results: list[EvalResult], output_dir: str | Path) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            result.serialize_to(out_dir)
