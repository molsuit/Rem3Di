from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.descriptor_analysis.analysis_tasks import (
    DescriptorAnalysisTask,
)
from remedi.evaluation.descriptor_analysis.context import (
    DescriptorAnalysisContext,
    DescriptorNormalizationConfig,
    ProjectionConfig,
)
from remedi.evaluation.results import EvalResult

logger = logging.getLogger(__name__)


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
        output_dir: str | Path | None = None,
        sample_indices: np.ndarray | None = None,
    ) -> list[EvalResult]:
        """Run every task; if ``output_dir`` is set, each task's artifacts are
        serialized as soon as it finishes — so a timeout or crash mid-pipeline
        still leaves the completed analyses on disk.

        Pass ``sample_indices`` when ``descriptors`` were computed on a subset
        of ``dataset`` so tasks that look up structures (e.g. TopNorm) can map
        descriptor rows back to original dataset indices.
        """
        ctx = DescriptorAnalysisContext.build(
            dataset=dataset,
            descriptors=descriptors,
            normalization=self.normalization,
            projection_config=self.projection,
            file_prefix=self.file_prefix,
            sample_indices=sample_indices,
        )

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        results: list[EvalResult] = []
        for task in self.tasks:
            task_label = type(task).__name__
            try:
                task_results = task.run(ctx)
                if out_dir is not None:
                    for result in task_results:
                        result.serialize_to(out_dir)
                results.extend(task_results)
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
