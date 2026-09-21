"""Run a prepare manifest (bundles -> conformers -> zarrs) fault-tolerantly.

One manifest describes one prepare run over a panel of benchmark bundles.
Tasks are expanded to one per dataset, so a dataset that fails is recorded in
``status.yaml`` while the rest of the panel still finishes.

Replaces ``build_benchmark_dataset.py`` (``BENCHMARK_DATA_FORMAT.md`` §3).

Usage::

    uv run python scripts/dataset_creation/prepare_benchmarks.py \
        --config configs/dataset_creation/prepare_benchmarks_template.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from remedi.data_handling.prepare import PrepareManifest, prepare

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    manifest = pyd_yaml.parse_yaml_file_as(PrepareManifest, args.config)
    report = prepare(manifest)
    logger.info(
        "prepare done: %d/%d task(s) ok -> %s",
        report.n_tasks - report.n_failed,
        report.n_tasks,
        manifest.output_root,
    )
    if report.n_failed:
        raise SystemExit(f"{report.n_failed} task(s) failed; see status.yaml")


if __name__ == "__main__":
    main()
