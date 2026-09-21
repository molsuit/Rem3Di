"""The runnable-task protocol + per-task status record.

A task is any object with a ``kind`` string and a ``run(ctx)`` that yields
:class:`EvalResult` artifacts. Concrete tasks are pydantic config models (so a
manifest stays declarative yaml) that also implement ``run`` — the same
config-is-the-task pattern as ``latent_evaluation.analysis_tasks``.

The protocol is deliberately **context-agnostic**: an eval task is handed an
``EvalContext`` and a prepare task a ``PrepareContext`` (``BENCHMARK_DATA_FORMAT.md``
§3 — the two contexts share the runner, not a base class). Typing ``ctx`` as
``Any`` is what lets :mod:`remedi.evaluation.framework.task_runner` stay a
dependency-light leaf that both sides import.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from remedi.evaluation.results import EvalResult


@runtime_checkable
class RunnableTask(Protocol):
    # Read-only on purpose: every concrete task narrows it to a
    # ``Literal["..."]`` discriminator, which a writable ``kind: str`` member
    # would reject for invariance.
    @property
    def kind(self) -> str: ...

    def run(self, ctx: Any) -> Iterator[EvalResult]: ...


def task_status_name(task: Any, index: int) -> str:
    """The ``status.yaml`` entry name for the ``index``-th task.

    ``<index>_<kind>``, with the task's own ``status_label`` appended when it
    defines one. A prepare manifest expands one task per dataset, and the label
    is what makes ``status.yaml`` answer "which of the 30 built" without
    reading a traceback.
    """
    label = getattr(task, "status_label", None)
    return f"{index}_{task.kind}_{label}" if label else f"{index}_{task.kind}"


class TaskStatus(BaseModel):
    """One task's outcome, recorded in the run ``status.yaml``."""

    name: str
    kind: str
    ok: bool
    duration_s: float
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
