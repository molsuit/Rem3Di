"""Unified, fault-tolerant evaluation framework.

One declarative :class:`EvalManifest` (model x output root x tasks) is executed
by :func:`run_manifest`, which serialises artifacts incrementally, shares
expensive resources across tasks via a :class:`ResourceCache`, and always writes
``manifest.yaml`` + ``status.yaml``. Plotting is decoupled (see
:mod:`framework.plotting`): tasks emit data artifacts, a registry renders them.
"""

from remedi_data.runs.task import RunnableTask, TaskStatus
from remedi_data.runs.task_runner import RunReport, run_tasks
from remedi_eval.framework.config import EvalManifest, TaskConfig
from remedi_eval.framework.context import EvalContext
from remedi_eval.framework.plotting import register_plotter, render
from remedi_eval.framework.resources import (
    EmbeddingSpec,
    ResourceCache,
    ResourceSpec,
)
from remedi_eval.framework.runner import run_manifest
from remedi_eval.framework.tasks import BenchmarkPanelConfig

__all__ = [
    "BenchmarkPanelConfig",
    "EmbeddingSpec",
    "EvalContext",
    "EvalManifest",
    "ResourceCache",
    "ResourceSpec",
    "RunReport",
    "RunnableTask",
    "TaskConfig",
    "TaskStatus",
    "register_plotter",
    "render",
    "run_manifest",
    "run_tasks",
]
