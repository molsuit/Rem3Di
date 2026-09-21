"""Fault-tolerant eval-manifest runner.

Builds the run's :class:`EvalContext`, hands the manifest's tasks to the shared
:func:`remedi.evaluation.framework.task_runner.run_tasks` loop, and always
writes ``manifest.yaml`` (resolved config) and ``status.yaml`` (the
:class:`RunReport`) so the run is self-describing even on failure.

The per-task loop itself lives in :mod:`task_runner`, which the prepare
manifest (``BENCHMARK_DATA_FORMAT.md`` §3) shares.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from remedi.evaluation.framework.config import EvalManifest
from remedi.evaluation.framework.context import EvalContext
from remedi.evaluation.framework.resources import ResourceCache
from remedi.evaluation.framework.task import TaskStatus
from remedi.evaluation.framework.task_runner import (
    RunReport,
    make_run_report,
    run_tasks,
)

logger = logging.getLogger(__name__)

__all__ = ["RunReport", "run_manifest"]


def run_manifest(manifest: EvalManifest) -> RunReport:
    """Run every task of ``manifest`` into ``manifest.output_root``."""
    out = Path(manifest.output_root)
    out.mkdir(parents=True, exist_ok=True)
    ctx = EvalContext(
        output_root=out,
        resource_cache_dir=manifest.cache_dir(),
        resources=ResourceCache(),
        model=manifest.model,
        seed=manifest.seed,
    )

    # ``run_tasks`` re-raises on ``keep_going=False``; mirroring its statuses
    # into this list keeps the outer ``finally`` writing the partial run rather
    # than overwriting ``status.yaml`` with an empty one.
    statuses: list[TaskStatus] = []

    def flush(current: list[TaskStatus]) -> None:
        statuses[:] = current
        _write_run_files(out, manifest, statuses)

    try:
        run_tasks(
            manifest.tasks,
            ctx,
            out,
            keep_going=manifest.keep_going,
            flush=flush,
        )
    finally:
        _write_run_files(out, manifest, statuses)

    report = make_run_report(len(manifest.tasks), statuses)
    if report.n_failed:
        logger.warning("%d/%d task(s) failed", report.n_failed, len(statuses))
    return report


def _write_run_files(
    out: Path, manifest: EvalManifest, statuses: list[TaskStatus]
) -> None:
    pyd_yaml.to_yaml_file(out / "manifest.yaml", manifest)
    pyd_yaml.to_yaml_file(
        out / "status.yaml", make_run_report(len(manifest.tasks), statuses)
    )
