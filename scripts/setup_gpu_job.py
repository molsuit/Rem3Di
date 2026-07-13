"""Stage a dataset onto the compute node's fast local scratch.

The staged path is printed on stdout so an sbatch script can capture it via
``$(...)`` and pass it as ``--dataset_path`` to the entrypoint; informational
messages go to stderr.

Staging directory resolution:

* Isambard-AI exposes a per-job tmpfs via ``$LOCALDIR`` (≈ ``/local/user/<UID>``,
  ~48 GiB). When set, it is used.
* JUWELS Booster nodes have no ``$LOCALDIR`` but do have a large RAM-backed
  ``/dev/shm`` shared by the (up to four) GPU processes packed onto the node, so
  that is the fallback.
* ``--stage-dir`` overrides both.

Staging is idempotent: if the target already exists it is reused, so the four
single-GPU runs packed onto a Booster node that share a ``dataset_path`` only
copy it once.

The input yaml only needs a top-level ``dataset_path`` field. Both
``TrainingConfig`` and ``ProfileBenchmarkConfig`` satisfy that — no pydantic
validation is performed here so this stager is config-shape-agnostic.

Usage from sbatch::

    LOCAL_DATASET_PATH=$(uv run python scripts/setup_gpu_job.py "$CFG")
    uv run python scripts/run_*.py --config "$CFG" \\
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


def default_stage_dir() -> Path:
    """Node-local fast scratch: ``$LOCALDIR`` (Isambard) or ``/dev/shm`` (JUWELS)."""
    value = os.environ.get("LOCALDIR")
    if value:
        return Path(value)
    return Path("/dev/shm")


def stage_dataset(source: Path, stage_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / source.name
    if target.exists():
        log(f"Reusing already-staged dataset: {target}")
        return target
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
        raise KeyError(f"{config_path} does not have a top-level 'dataset_path' field.")
    return Path(str(data["dataset_path"]).strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a dataset to node-local scratch ($LOCALDIR or /dev/shm)."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a yaml with a top-level dataset_path field.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Override the node-local staging directory "
        "(default: $LOCALDIR if set, else /dev/shm).",
    )
    args = parser.parse_args()

    src = read_dataset_path(args.config)
    stage_dir = args.stage_dir if args.stage_dir is not None else default_stage_dir()
    staged = stage_dataset(src, stage_dir)
    print(staged)


if __name__ == "__main__":
    main()
