"""Fault-tolerant manifest runner.

Runs each task in the manifest, serialising artifacts as they are yielded so a
crash / wall-clock timeout keeps everything completed so far. Every task is
wrapped: failures are recorded (with traceback) in ``status.yaml`` rather than
aborting the run (unless ``keep_going=False``). A ``finally`` block always
writes ``manifest.yaml`` (resolved config + git SHA) and ``status.yaml`` so the
run is self-describing even on failure.
"""

from __future__ import annotations

import logging
import subprocess
import time
import traceback
from pathlib import Path

import pydantic_yaml as pyd_yaml
from pydantic import BaseModel

from threedscriptors.evaluation.framework.config import EvalManifest
from threedscriptors.evaluation.framework.context import EvalContext
from threedscriptors.evaluation.framework.resources import ResourceCache
from threedscriptors.evaluation.framework.task import TaskStatus

logger = logging.getLogger(__name__)


class RunReport(BaseModel):
    git_sha: str | None
    n_tasks: int
    n_failed: int
    statuses: list[TaskStatus]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # git missing / not a repo — provenance is best-effort
        return None


def run_manifest(manifest: EvalManifest) -> RunReport:
    out = Path(manifest.output_root)
    out.mkdir(parents=True, exist_ok=True)
    ctx = EvalContext(
        output_root=out,
        resource_cache_dir=manifest.cache_dir(),
        resources=ResourceCache(),
        model=manifest.model,
        seed=manifest.seed,
    )

    statuses: list[TaskStatus] = []
    try:
        for i, task in enumerate(manifest.tasks):
            name = f"{i}_{task.kind}"
            logger.info("=== task %s ===", name)
            t0 = time.monotonic()
            try:
                artifacts = [art.serialize_to(out) for art in task.run(ctx)]
                statuses.append(
                    TaskStatus(
                        name=name,
                        kind=task.kind,
                        ok=True,
                        duration_s=time.monotonic() - t0,
                        artifacts=artifacts,
                    )
                )
            except Exception as exc:  # one task must not sink the run
                logger.exception("task %s failed", name)
                statuses.append(
                    TaskStatus(
                        name=name,
                        kind=task.kind,
                        ok=False,
                        duration_s=time.monotonic() - t0,
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                )
                if not manifest.keep_going:
                    raise
            finally:
                # Flush after every task so partial runs are inspectable.
                _write_run_files(out, manifest, statuses)
    finally:
        _write_run_files(out, manifest, statuses)

    n_failed = sum(not s.ok for s in statuses)
    if n_failed:
        logger.warning("%d/%d task(s) failed", n_failed, len(statuses))
    return RunReport(
        git_sha=_git_sha(),
        n_tasks=len(manifest.tasks),
        n_failed=n_failed,
        statuses=statuses,
    )


def _write_run_files(
    out: Path, manifest: EvalManifest, statuses: list[TaskStatus]
) -> None:
    pyd_yaml.to_yaml_file(out / "manifest.yaml", manifest)
    report = RunReport(
        git_sha=_git_sha(),
        n_tasks=len(manifest.tasks),
        n_failed=sum(not s.ok for s in statuses),
        statuses=statuses,
    )
    pyd_yaml.to_yaml_file(out / "status.yaml", report)
