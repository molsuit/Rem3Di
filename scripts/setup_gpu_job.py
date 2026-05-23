"""Stage a dataset onto the compute node's fast local scratch.

On Isambard-AI compute nodes ``$LOCALDIR`` (≈ ``/local/user/<UID>``) is a
per-job tmpfs of ~48 GiB. Reading the dataset from there avoids hammering
shared storage during training/benchmarking.

The staged path is printed on stdout so an sbatch script can capture it via
``$(...)`` and pass it as ``--dataset_path`` to the entrypoint; informational
messages go to stderr.

The input yaml only needs a top-level ``dataset_path`` field. Both
``TrainingConfig`` and ``ProfileBenchmarkConfig`` satisfy that — no pydantic
validation is performed here so this stager is config-shape-agnostic.

Usage from sbatch::

    LOCAL_DATASET_PATH=$(uv run scripts/setup_gpu_job.py "$CFG")
    srun uv run scripts/run_*.py --config "$CFG" \\
                                 --dataset_path "$LOCAL_DATASET_PATH"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


def log(msg: str) -> None:
    print(f"[setup_gpu_job] {msg}", file=sys.stderr, flush=True)


def localdir() -> Path:
    value = os.environ.get("LOCALDIR")
    if not value:
        raise OSError(
            "$LOCALDIR is not set; this helper assumes Isambard-AI environment."
        )
    return Path(value)


def stage_dataset(source: Path, stage_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / source.name
    log(f"Staging {source} -> {target}")
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
    log("Staging complete.")
    return target


def read_dataset_path(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "dataset_path" not in data:
        raise KeyError(
            f"{config_path} does not have a top-level 'dataset_path' field."
        )
    return Path(str(data["dataset_path"]).strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a dataset to $LOCALDIR on Isambard-AI."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a yaml with a top-level dataset_path field.",
    )
    args = parser.parse_args()

    src = read_dataset_path(args.config)
    staged = stage_dataset(src, localdir())
    print(staged)


if __name__ == "__main__":
    main()
