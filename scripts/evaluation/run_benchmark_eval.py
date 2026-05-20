"""Run the descriptor-probe benchmark eval from a yaml config.

Walks ``eval_root`` (one prepared zarr per benchmark, with a sidecar
manifest), runs the descriptor x learner cross-product, and writes
``results.csv`` + ``results.yaml`` under ``output_dir``.

Usage::

    uv run python scripts/evaluation/run_benchmark_eval.py \\
        --config configs/eval/benchmark_eval.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from threedscriptors.evaluation.benchmark.runner import EvalConfig, run_eval

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = pyd_yaml.parse_yaml_file_as(EvalConfig, args.config)
    rows = run_eval(cfg)
    logger.info("benchmark eval: %d result rows written to %s", len(rows), cfg.output_dir)


if __name__ == "__main__":
    main()
