"""Fault-tolerant prepare-manifest runner (§3).

Mirrors :func:`remedi.evaluation.framework.runner.run_manifest`: it builds the
run's context, hands the *expanded* task list to the shared
:func:`remedi.evaluation.framework.task_runner.run_tasks` loop, and always
writes ``manifest.yaml`` and ``status.yaml`` under ``output_root``.

The gain over the script it replaces is exactly the framework's: fault
isolation per dataset and a ``status.yaml`` that answers "which of the 30
built" in one file instead of scrollback. That matters most for
``generate_conformers``, the long pole — a wall-clock timeout mid-panel must
not lose the datasets that finished.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from remedi.data_handling.prepare.config import PrepareManifest, PrepareTask
from remedi.data_handling.prepare.context import PrepareContext
from remedi.evaluation.framework.task import TaskStatus
from remedi.evaluation.framework.task_runner import (
    RunReport,
    make_run_report,
    run_tasks,
)

logger = logging.getLogger(__name__)


def prepare(manifest: PrepareManifest) -> RunReport:
    """Run every expanded task of ``manifest`` into ``manifest.output_root``."""
    out = Path(manifest.output_root)
    out.mkdir(parents=True, exist_ok=True)
    Path(manifest.benchmark_root).mkdir(parents=True, exist_ok=True)
    ctx = PrepareContext(
        smiles_bundle_root=Path(manifest.smiles_bundle_root),
        benchmark_root=Path(manifest.benchmark_root),
        output_root=out,
        seed=manifest.seed,
    )

    tasks = manifest.expand_tasks()
    # ``run_tasks`` re-raises on ``keep_going=False``; mirroring its statuses
    # into this list keeps the outer ``finally`` writing the partial run rather
    # than overwriting ``status.yaml`` with an empty one.
    statuses: list[TaskStatus] = []

    def flush(current: list[TaskStatus]) -> None:
        statuses[:] = current
        _write_run_files(out, manifest, tasks, statuses)

    try:
        run_tasks(tasks, ctx, out, keep_going=manifest.keep_going, flush=flush)
    finally:
        _write_run_files(out, manifest, tasks, statuses)

    report = make_run_report(len(tasks), statuses)
    if report.n_failed:
        logger.warning("%d/%d task(s) failed", report.n_failed, len(statuses))
    return report


def _write_run_files(
    out: Path,
    manifest: PrepareManifest,
    tasks: list[PrepareTask],
    statuses: list[TaskStatus],
) -> None:
    pyd_yaml.to_yaml_file(out / "manifest.yaml", manifest)
    pyd_yaml.to_yaml_file(out / "status.yaml", make_run_report(len(tasks), statuses))
