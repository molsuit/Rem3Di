"""Snapshot pretraining configs into their run directories and submit one DAIS job each.

Each config is frozen at submission time (with its architecture config), so
queued jobs are not affected by later edits to the source yaml files, and its
``training_directory`` is created so the running job writes its outputs next to
the snapshot. Every run is one ``scripts/pretrain.slurm`` job on one H200 or B200.

Usage::

    uv run python scripts/submit_job.py cfgA.yaml cfgB.yaml --group scaling --time 06:00:00

Set ``--time`` to the expected runtime plus a modest margin: a job whose limit
fits a gap in the schedule starts early (backfill), a 24 h default waits.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pydantic_yaml as pyaml

from remedi.configuration.training_config import PretrainingConfig


def log(msg: str) -> None:
    print(f"[submit_job] {msg}", file=sys.stderr, flush=True)


def snapshot_configs(training_config_path: Path) -> tuple[Path, str]:
    training_config = pyaml.parse_yaml_file_as(PretrainingConfig, training_config_path)
    # A fresh numbered run directory, also when the config names one.
    training_config.training_directory = None
    training_directory = training_config.create_run_directory()

    architecture_snapshot = training_directory / "architecture_config.yaml"
    shutil.copyfile(training_config.model_config_path, architecture_snapshot)

    training_config.model_config_path = architecture_snapshot
    training_config.training_directory = training_directory
    training_snapshot = training_directory / "training_config.yaml"
    pyaml.to_yaml_file(training_snapshot, training_config)

    log(f"Snapshot training directory: {training_directory}")
    return training_snapshot, training_config.training_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot pretraining configs into frozen training directories "
        "and submit one SLURM job per config."
    )
    parser.add_argument(
        "training_configs",
        type=Path,
        nargs="+",
        help="One or more training_config.yaml files (one per run).",
    )
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        default=Path("scripts/pretrain.slurm"),
        help="sbatch script to invoke (default: scripts/pretrain.slurm)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="pretrain",
        help="SLURM job-name prefix; each run becomes <group>_<training_name>.",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="SLURM time limit (HH:MM:SS); default: the sbatch script's 24 h.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Snapshot configs only; do not call sbatch.",
    )
    args = parser.parse_args()

    # Snapshot every config up front so the whole set is frozen together.
    snapshots = [snapshot_configs(config) for config in args.training_configs]

    for snapshot, name in snapshots:
        if args.no_submit:
            print(snapshot)
            continue
        command = [
            "sbatch",
            f"--job-name={args.group}_{name}",
            *([f"--time={args.time}"] if args.time else []),
            str(args.sbatch_script),
            str(snapshot),
        ]
        log(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
