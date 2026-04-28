"""Stage a training dataset onto the compute node's fast local scratch.

On Isambard-AI compute nodes ``$LOCALDIR`` (≈ ``/local/user/<UID>``) is a
per-job tmpfs of ~48 GiB. Reading the dataset from there avoids hammering
shared storage during training.

The staged path is printed on stdout so an sbatch script can capture it via
``$(...)`` and pass it as ``--dataset_path`` to the training entrypoint;
informational messages go to stderr.

Usage from sbatch::

    LOCAL_DATASET_PATH=$(uv run scripts/setup_gpu_job.py "$CFG")
    srun uv run scripts/run_*.py --training_config "$CFG" \\
                                 --dataset_path "$LOCAL_DATASET_PATH"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pydantic_yaml as pyaml

from threedscriptors.configuration.training_config import TrainingConfig


def log(msg: str) -> None:
    print(f"[setup_gpu_job] {msg}", file=sys.stderr, flush=True)


def localdir() -> Path:
    value = os.environ.get("LOCALDIR")
    if not value:
        raise EnvironmentError(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage training dataset to $LOCALDIR on Isambard-AI."
    )
    parser.add_argument("training_config", type=Path)
    args = parser.parse_args()

    cfg = pyaml.parse_yaml_file_as(TrainingConfig, args.training_config)
    staged = stage_dataset(cfg.dataset_path, localdir())
    print(staged)


if __name__ == "__main__":
    main()
