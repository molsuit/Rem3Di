"""Eval task protocol + per-task status record.

A task is any object with a ``kind`` string and a ``run(ctx)`` that yields
:class:`EvalResult` artifacts. Concrete tasks are pydantic config models (so a
manifest stays declarative yaml) that also implement ``run`` — the same
config-is-the-task pattern as ``descriptor_analysis.analysis_tasks``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from remedi.evaluation.framework.context import EvalContext
from remedi.evaluation.results import EvalResult


@runtime_checkable
class EvalTask(Protocol):
    kind: str

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]: ...


class TaskStatus(BaseModel):
    """One task's outcome, recorded in the run ``status.yaml``."""

    name: str
    kind: str
    ok: bool
    duration_s: float
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
