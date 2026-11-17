"""Fault-tolerant eval-manifest runner.

Builds the run's :class:`EvalContext` and hands the manifest's tasks to the
shared :func:`remedi_data.runs.task_runner.run_and_record`, which writes
``manifest.yaml`` (resolved config) first and ``status.yaml`` (the
:class:`RunReport`) after every task, so the run is self-describing even on
failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from remedi_data.runs.task_runner import (
    RunReport,
    run_and_record,
)
from remedi_eval.framework.config import EvalManifest
from remedi_eval.framework.context import EvalContext

logger = logging.getLogger(__name__)

__all__ = ["RunReport", "run_manifest"]


def run_manifest(manifest: EvalManifest) -> RunReport:
    """Run every task of ``manifest`` into ``manifest.output_root``."""
    out = Path(manifest.output_root)
    out.mkdir(parents=True, exist_ok=True)
    ctx = EvalContext(
        output_root=out,
        descriptor_cache_dir=manifest.descriptor_cache(),
        model=manifest.model,
        seed=manifest.seed,
        keep_going=manifest.keep_going,
    )

    return run_and_record(
        [manifest.tasks], ctx, out, manifest, keep_going=manifest.keep_going
    )
