"""The shared, fault-tolerant per-task loop (``BENCHMARK_DATA_FORMAT.md`` §3).

Both manifests run through here: :func:`remedi.evaluation.framework.runner.run_manifest`
with an ``EvalContext`` and :func:`remedi.data_handling.prepare.runner.prepare`
with a ``PrepareContext``. The two contexts share the runner, not a base class,
so this module knows nothing about either — it takes whatever ``ctx`` the caller
built and hands it to ``task.run``.

This module is a dependency-light leaf on purpose: it imports stdlib, pydantic
and :mod:`remedi.evaluation.framework.task` only, so importing it from
``remedi.data_handling.prepare`` does not drag in descriptors, torch or MACE.
"""

from __future__ import annotations

import logging
import subprocess
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from remedi.evaluation.framework.task import (
    RunnableTask,
    TaskStatus,
    task_status_name,
)

logger = logging.getLogger(__name__)


class RunReport(BaseModel):
    """What a whole manifest run did — serialised as ``status.yaml``."""

    git_sha: str | None
    n_tasks: int
    n_failed: int
    statuses: list[TaskStatus]


def git_sha() -> str | None:
    """The HEAD sha of the repository the process runs in, best effort."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except Exception:  # git missing / not a repo — provenance is best-effort
        return None


def run_tasks(
    tasks: Sequence[RunnableTask],
    ctx: Any,
    out: Path,
    *,
    keep_going: bool,
    flush: Callable[[list[TaskStatus]], None] | None = None,
    first_index: int = 0,
) -> list[TaskStatus]:
    """Run each task, serialising artifacts as they are yielded.

    Every task is wrapped: a failure is recorded (with traceback) in the
    returned :class:`TaskStatus` list rather than aborting the run, unless
    ``keep_going`` is False. ``flush`` — when given — is called with the
    statuses so far after *every* task, so a wall-clock timeout mid-run still
    leaves an inspectable ``status.yaml``.

    Args:
        tasks: the expanded task list; each needs ``kind`` and ``run(ctx)``.
        ctx: the per-run context handed to every task, unexamined here.
        out: the run's output root; artifacts serialise relative to it.
        keep_going: False re-raises the first failure after recording it.
        flush: called with the statuses so far after each task.
        first_index: the number the first task's ``status.yaml`` name counts
            from. A prepare run calls this once per manifest task, expanding
            each into its per-dataset tasks only when that group is about to
            run, so the offset is what keeps the entry names unique and
            monotonic across groups.

    Returns:
        One :class:`TaskStatus` per task that was started, in order.
    """
    statuses: list[TaskStatus] = []
    for offset, task in enumerate(tasks):
        name = task_status_name(task, first_index + offset)
        logger.info("=== task %s ===", name)
        started = time.monotonic()
        try:
            artifacts = [artifact.serialize_to(out) for artifact in task.run(ctx)]
            statuses.append(
                TaskStatus(
                    name=name,
                    kind=task.kind,
                    ok=True,
                    duration_s=time.monotonic() - started,
                    artifacts=artifacts,
                )
            )
        except Exception as exception:  # one task must not sink the run
            logger.exception("task %s failed", name)
            statuses.append(
                TaskStatus(
                    name=name,
                    kind=task.kind,
                    ok=False,
                    duration_s=time.monotonic() - started,
                    error=repr(exception),
                    traceback=traceback.format_exc(),
                )
            )
            if not keep_going:
                raise
        finally:
            # Flush after every task so partial runs are inspectable.
            if flush is not None:
                flush(statuses)
    return statuses


def make_run_report(n_tasks: int, statuses: list[TaskStatus]) -> RunReport:
    """Assemble the report written as ``status.yaml`` and returned to the caller."""
    return RunReport(
        git_sha=git_sha(),
        n_tasks=n_tasks,
        n_failed=sum(not status.ok for status in statuses),
        statuses=statuses,
    )
