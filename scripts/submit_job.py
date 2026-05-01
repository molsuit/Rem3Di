"""Snapshot training + architecture configs into a per-submission directory and submit to SLURM.

Freezes the configs at submission time so queued jobs are not affected by later
edits to the source YAML files. Also pre-creates ``training_directory`` so the
running job writes its outputs alongside the snapshot.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pydantic_yaml as pyaml

from threedscriptors.configuration.training_config import TrainingConfig


def log(msg: str) -> None:
    print(f"[submit_job] {msg}", file=sys.stderr, flush=True)


def snapshot_configs(training_config_path: Path) -> Path:
    training_config = pyaml.parse_yaml_file_as(TrainingConfig, training_config_path)

    output_base = training_config.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    training_idx = len(list(output_base.glob("*/")))
    now = datetime.now()
    training_directory = output_base / (
        f"{training_idx}-{now.strftime('%Y_%m_%d_%H_%M_%S')}-{training_config.training_name}"
    )
    training_directory.mkdir()

    architecture_snapshot = training_directory / "architecture_config.yaml"
    shutil.copyfile(training_config.model_config_path, architecture_snapshot)

    training_config.model_config_path = architecture_snapshot
    training_config.training_directory = training_directory
    training_snapshot = training_directory / "training_config.yaml"
    pyaml.to_yaml_file(training_snapshot, training_config)

    log(f"Snapshot training directory: {training_directory}")
    return training_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot configs into a frozen training directory and submit to SLURM."
    )
    parser.add_argument("training_config", type=Path)
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        default=Path("submit_training_run.sh"),
        help="sbatch script to invoke (default: submit_training_run.sh)",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Snapshot configs only; do not call sbatch.",
    )
    args = parser.parse_args()

    training_snapshot = snapshot_configs(args.training_config)

    if args.no_submit:
        print(training_snapshot)
        return

    cmd = ["sbatch", str(args.sbatch_script), str(training_snapshot)]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
