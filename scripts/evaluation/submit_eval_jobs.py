"""Submit per-model eval config directories packed four-per-node to SLURM.

Takes one or more *eval config directories* (each holding a unified
``manifest.yaml``, as written by ``generate_pcqm_novicreg_eval_configs.py``).
JUWELS Booster only allocates whole nodes (4x A100), so the model evals are
packed four-per-node: the directories are grouped into batches of
``--batch-size`` (default 4) and one ``submit_eval_node.sbatch`` job is submitted
per batch, running one model eval per GPU.

Usage::

    uv run python scripts/evaluation/submit_eval_jobs.py \\
        configs/eval/pcqm_ablation_novicreg/*
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED = ("manifest.yaml",)


def log(msg: str) -> None:
    print(f"[submit_eval] {msg}", file=sys.stderr, flush=True)


def validate(config_dir: Path) -> Path:
    if not config_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {config_dir}")
    missing = [f for f in REQUIRED if not (config_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{config_dir} is missing {missing}; regenerate with "
            "generate_pcqm_novicreg_eval_configs.py"
        )
    return config_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit eval config dirs packed N-per-node to SLURM."
    )
    parser.add_argument(
        "config_dirs",
        type=Path,
        nargs="+",
        help="One or more eval config directories (each with a manifest.yaml).",
    )
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        default=Path("scripts/evaluation/submit_eval_node.sbatch"),
        help="sbatch script to invoke (default: scripts/evaluation/submit_eval_node.sbatch)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="eval",
        help="SLURM job-name prefix; each batch becomes <group>_b<N>.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Model evals (GPUs) per node. JUWELS Booster nodes have 4 (default: 4).",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="List the batches only; do not call sbatch.",
    )
    args = parser.parse_args()

    dirs = [validate(d) for d in args.config_dirs]
    batches = [
        dirs[i : i + args.batch_size]
        for i in range(0, len(dirs), args.batch_size)
    ]
    log(
        f"{len(dirs)} model eval(s) -> {len(batches)} node job(s) "
        f"of up to {args.batch_size}."
    )

    for b_idx, batch in enumerate(batches):
        if args.no_submit:
            log(f"[batch {b_idx}] (no-submit) config dirs:")
            for d in batch:
                print(d)
            continue
        cmd = [
            "sbatch",
            f"--job-name={args.group}_b{b_idx}",
            str(args.sbatch_script),
            *[str(d) for d in batch],
        ]
        log(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
