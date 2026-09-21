"""Fault-tolerant prepare-manifest runner (§3).

Mirrors :func:`remedi.evaluation.framework.runner.run_manifest`: it builds the
run's context, hands the tasks to the shared
:func:`remedi.evaluation.framework.task_runner.run_tasks` loop, and always
writes ``manifest.yaml`` and ``status.yaml`` under ``output_root``.

The gain over the script it replaces is exactly the framework's: fault
isolation per dataset and a ``status.yaml`` that answers "which of the 30
built" in one file instead of scrollback. That matters most for
``generate_conformers``, the long pole — a wall-clock timeout mid-panel must
not lose the datasets that finished.

Unlike the eval runner it runs the manifest **group by group**: each manifest
task is expanded into its per-dataset tasks only once every earlier group has
finished. A prepare manifest is a pipeline — ``generate_conformers`` writes the
very bundles that ``ingest_benchmark`` then discovers — so resolving
``dataset_ids: null`` for the whole manifest up front sees a directory that the
run has not filled in yet. On the 22-endpoint TDC panel that produced 22 + 7 + 7
entries instead of 66.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from remedi.data_handling.prepare.config import PrepareManifest
from remedi.data_handling.prepare.context import PrepareContext
from remedi.evaluation.framework.task import TaskStatus
from remedi.evaluation.framework.task_runner import (
    RunReport,
    make_run_report,
    run_tasks,
)

logger = logging.getLogger(__name__)


def prepare(manifest: PrepareManifest) -> RunReport:
    """Run every task of ``manifest``, per dataset, into ``manifest.output_root``.

    Raises:
        Exception: whatever a task raised, when ``manifest.keep_going`` is
            False. ``manifest.yaml`` and ``status.yaml`` are written first, so
            the partial run stays inspectable.
    """
    out = Path(manifest.output_root)
    out.mkdir(parents=True, exist_ok=True)
    Path(manifest.benchmark_root).mkdir(parents=True, exist_ok=True)
    ctx = PrepareContext(
        smiles_bundle_root=Path(manifest.smiles_bundle_root),
        benchmark_root=Path(manifest.benchmark_root),
        output_root=out,
        seed=manifest.seed,
    )

    # ``run_tasks`` re-raises on ``keep_going=False``; mirroring its statuses
    # into ``running`` keeps the ``finally`` below writing the partial run
    # rather than overwriting ``status.yaml`` with an empty one.
    completed: list[TaskStatus] = []
    running: list[TaskStatus] = []
    # Tasks whose group has already been expanded. Groups further down the
    # manifest are not resolved yet, so this grows as the run proceeds.
    planned = 0

    def flush(current: list[TaskStatus]) -> None:
        running[:] = current
        _write_run_files(out, manifest, planned, [*completed, *running])

    for group_index, task in enumerate(manifest.tasks):
        group = manifest.expand_task(task)
        planned += len(group)
        running.clear()
        logger.info(
            "=== manifest task %d (%s): %d dataset(s) ===",
            group_index,
            task.kind,
            len(group),
        )
        try:
            run_tasks(
                group,
                ctx,
                out,
                keep_going=manifest.keep_going,
                flush=flush,
                first_index=len(completed),
            )
        finally:
            completed.extend(running)
            running.clear()
            _write_run_files(out, manifest, planned, completed)

    report = make_run_report(planned, completed)
    if report.n_failed:
        logger.warning("%d/%d task(s) failed", report.n_failed, len(completed))
    return report


def _write_run_files(
    out: Path,
    manifest: PrepareManifest,
    n_tasks: int,
    statuses: list[TaskStatus],
) -> None:
    pyd_yaml.to_yaml_file(out / "manifest.yaml", manifest)
    pyd_yaml.to_yaml_file(out / "status.yaml", make_run_report(n_tasks, statuses))
