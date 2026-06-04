"""Run a unified evaluation manifest (model x tasks) fault-tolerantly.

One manifest describes one model's full evaluation — a benchmark panel, a
retrieval panel, etc. Tasks share embeddings / indices via the run's
:class:`ResourceCache`; artifacts are written incrementally and a failing task
is recorded in ``status.yaml`` rather than sinking the run.

This is the single eval entrypoint — it superseded the former separate
benchmark / retrieval runner scripts.

Usage::

    uv run python scripts/evaluation/run_eval.py --config <manifest>.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from threedscriptors.evaluation.framework import EvalManifest, run_manifest

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    manifest = pyd_yaml.parse_yaml_file_as(EvalManifest, args.config)
    report = run_manifest(manifest)
    logger.info(
        "eval done: %d/%d task(s) ok -> %s",
        report.n_tasks - report.n_failed,
        report.n_tasks,
        manifest.output_root,
    )
    if report.n_failed:
        raise SystemExit(f"{report.n_failed} task(s) failed; see status.yaml")


if __name__ == "__main__":
    main()
