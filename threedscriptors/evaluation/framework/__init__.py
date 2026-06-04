"""Unified, fault-tolerant evaluation framework.

One declarative :class:`EvalManifest` (model x output root x tasks) is executed
by :func:`run_manifest`, which serialises artifacts incrementally, shares
expensive resources across tasks via a :class:`ResourceCache`, and always writes
``manifest.yaml`` + ``status.yaml``. Plotting is decoupled (see
:mod:`framework.plotting`): tasks emit data artifacts, a registry renders them.
"""

from threedscriptors.evaluation.framework.config import EvalManifest, TaskConfig
from threedscriptors.evaluation.framework.context import EvalContext
from threedscriptors.evaluation.framework.plotting import register_plotter, render
from threedscriptors.evaluation.framework.resources import (
    EmbeddingSpec,
    FingerprintSpec,
    IndexSpec,
    ResourceCache,
    ResourceSpec,
)
from threedscriptors.evaluation.framework.runner import RunReport, run_manifest
from threedscriptors.evaluation.framework.task import EvalTask, TaskStatus
from threedscriptors.evaluation.framework.tasks import (
    BenchmarkPanelConfig,
    DescriptorAnalysisConfig,
    RetrievalConfig,
)

__all__ = [
    "BenchmarkPanelConfig",
    "DescriptorAnalysisConfig",
    "EmbeddingSpec",
    "EvalContext",
    "EvalManifest",
    "EvalTask",
    "FingerprintSpec",
    "IndexSpec",
    "ResourceCache",
    "ResourceSpec",
    "RetrievalConfig",
    "RunReport",
    "TaskConfig",
    "TaskStatus",
    "register_plotter",
    "render",
    "run_manifest",
]
