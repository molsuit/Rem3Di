"""Unified, fault-tolerant evaluation framework.

One declarative :class:`EvalManifest` (model x output root x tasks) is executed
by :func:`run_manifest`, which serialises artifacts incrementally, shares
expensive resources across tasks via a :class:`ResourceCache`, and always writes
``manifest.yaml`` + ``status.yaml``. Plotting is decoupled (see
:mod:`framework.plotting`): tasks emit data artifacts, a registry renders them.
"""

from remedi.evaluation.framework.config import EvalManifest, TaskConfig
from remedi.evaluation.framework.context import EvalContext
from remedi.evaluation.framework.plotting import register_plotter, render
from remedi.evaluation.framework.resources import (
    EmbeddingSpec,
    FingerprintSpec,
    IndexSpec,
    ResourceCache,
    ResourceSpec,
)
from remedi.evaluation.framework.runner import run_manifest
from remedi.evaluation.framework.task import RunnableTask, TaskStatus
from remedi.evaluation.framework.task_runner import RunReport, run_tasks
from remedi.evaluation.framework.tasks import (
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
    "FingerprintSpec",
    "IndexSpec",
    "ResourceCache",
    "ResourceSpec",
    "RetrievalConfig",
    "RunReport",
    "RunnableTask",
    "TaskConfig",
    "TaskStatus",
    "register_plotter",
    "render",
    "run_manifest",
    "run_tasks",
]
