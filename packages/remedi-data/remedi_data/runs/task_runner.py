"""The shared, fault-tolerant per-task loop (``BENCHMARK_DATA_FORMAT.md`` §3.2, §6).

Every run goes through :func:`run_and_record`:
:func:`remedi_eval.framework.runner.run_manifest` with an ``EvalContext``,
:func:`remedi_data.build.build_datasets` with its build and verify tasks and
:func:`remedi_data.preparation.structures.prepare.prepare_structure_datasets` with one task
per source. The contexts share the runner, not a base class, so this module
knows nothing about either: it takes whatever ``ctx`` the caller built and hands
it to ``task.run``.

It lives in ``remedi_data`` so the dataset build does not depend on the
evaluation code; it imports stdlib, pydantic and :mod:`remedi_data` only.
"""

from __future__ import annotations

import functools
import logging
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from remedi_data.format import git_head_sha
from remedi_data.runs.task import (
    RunnableTask,
    TaskStatus,
    task_status_name,
)
from remedi_data.yaml_io import write_model

logger = logging.getLogger(__name__)


class RunReport(BaseModel):
    """What a whole manifest run did — serialised as ``status.yaml``."""

    git_sha: str | None
    n_tasks: int
    n_failed: int
    statuses: list[TaskStatus]


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
            # A task may rewrite one artifact as it goes (a results table after
            # every cell); it is listed once, in the place it first appeared.
            artifacts_by_file: dict[str, dict[str, Any]] = {}
            for artifact in task.run(ctx):
                entry = artifact.serialize_to(out)
                artifacts_by_file[str(entry.get("file_name"))] = entry
            artifacts = list(artifacts_by_file.values())
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


@functools.cache
def repository_git_sha() -> str | None:
    """The git sha of the checkout this package is imported from, resolved once.

    ``status.yaml`` is rewritten after every task, so git runs once per process,
    not once per write; and it names the code that ran, not whatever repository
    the working directory happens to be in.
    """
    return git_head_sha(Path(__file__).parent)


def make_run_report(n_tasks: int, statuses: list[TaskStatus]) -> RunReport:
    """Assemble the report written as ``status.yaml`` and returned to the caller."""
    return RunReport(
        git_sha=repository_git_sha(),
        n_tasks=n_tasks,
        n_failed=sum(not status.ok for status in statuses),
        statuses=statuses,
    )


#: Written once, before the first task: the resolved config of the run.
MANIFEST_FILENAME = "manifest.yaml"
#: Rewritten after every task and once at the end: the :class:`RunReport`.
STATUS_FILENAME = "status.yaml"


def run_and_record(
    task_groups: Iterable[Sequence[RunnableTask]],
    context: Any,
    output_directory: Path,
    manifest: BaseModel,
    *,
    keep_going: bool,
) -> RunReport:
    """Run task groups in order and keep the run self-describing on disk.

    ``manifest.yaml`` (``manifest``) is written once, before the first task;
    ``status.yaml`` after every task and once more at the end, also when a task
    fails with ``keep_going`` False or a group cannot be built. A group is only
    taken from ``task_groups`` when the previous one has finished, so a
    generator can build its next group from what the previous one wrote (the
    dataset build verifies the datasets it has just built).

    Returns:
        The final :class:`RunReport` (the content of ``status.yaml``).
    """
    output_directory = Path(output_directory)
    write_model(output_directory / MANIFEST_FILENAME, manifest)
    finished: list[TaskStatus] = []
    # The statuses of the running group: ``run_tasks`` re-raises on
    # ``keep_going=False`` before returning them.
    running: list[TaskStatus] = []
    planned = 0

    def record() -> None:
        write_model(
            output_directory / STATUS_FILENAME,
            make_run_report(planned, [*finished, *running]),
        )

    def flush(current: list[TaskStatus]) -> None:
        running[:] = current
        record()

    try:
        for group in task_groups:
            planned += len(group)
            try:
                run_tasks(
                    group,
                    context,
                    output_directory,
                    keep_going=keep_going,
                    flush=flush,
                    first_index=len(finished),
                )
            finally:
                finished.extend(running)
                running.clear()
    finally:
        record()
    report = make_run_report(planned, finished)
    if report.n_failed:
        logger.warning("%d/%d task(s) failed", report.n_failed, planned)
    return report
