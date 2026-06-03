"""Snapshot training + architecture configs into per-submission directories and submit to SLURM.

Takes one or more training configs (one per ablation run). Each is frozen at
submission time so queued jobs are not affected by later edits to the source
YAML files, and its ``training_directory`` is pre-created so the running job
writes its outputs alongside the snapshot.

JUWELS Booster only allocates whole nodes (4x A100), so single-GPU runs are
packed four-per-node: the snapshots are grouped into batches of ``--batch-size``
(default 4) and one ``submit_ablation_node.sbatch`` job is submitted per batch.

Usage::

    uv run python scripts/submit_job.py cfgA.yaml cfgB.yaml cfgC.yaml cfgD.yaml
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


def snapshot_configs(training_config_path: Path) -> tuple[Path, str]:
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
    return training_snapshot, training_config.training_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot ablation configs into frozen training directories "
        "and submit them packed N-per-node to SLURM."
    )
    parser.add_argument(
        "training_configs",
        type=Path,
        nargs="+",
        help="One or more training_config.yaml files (one per ablation run).",
    )
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        default=Path("scripts/submit_ablation_node.sbatch"),
        help="sbatch script to invoke (default: scripts/submit_ablation_node.sbatch)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="ablation",
        help="SLURM job-name prefix; each batch becomes <group>_b<N>.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Runs (GPUs) per node. JUWELS Booster nodes have 4 (default: 4).",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Snapshot configs only; do not call sbatch.",
    )
    args = parser.parse_args()

    # Snapshot every config up front so the whole ablation is frozen together.
    snapshots = [snapshot_configs(cfg)[0] for cfg in args.training_configs]

    # Pack into batches -> one sbatch job (one node) per batch.
    batches = [
        snapshots[i : i + args.batch_size]
        for i in range(0, len(snapshots), args.batch_size)
    ]
    log(
        f"{len(snapshots)} run(s) -> {len(batches)} node job(s) "
        f"of up to {args.batch_size}."
    )

    for b_idx, batch in enumerate(batches):
        if args.no_submit:
            log(f"[batch {b_idx}] (no-submit) snapshots:")
            for snapshot in batch:
                print(snapshot)
            continue
        cmd = [
            "sbatch",
            f"--job-name={args.group}_b{b_idx}",
            str(args.sbatch_script),
            *[str(snapshot) for snapshot in batch],
        ]
        log(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
