"""Time Butina clustering on nested molecule samples of a dataset (§5.2.3, step 3).

For each sample size and coarse cutoff: coarse Butina time and cluster sizes; at
the first coarse cutoff also the fine level. A size is skipped when the quadratic
extrapolation from the previous one exceeds ``--budget-seconds``. Results go to
``--output`` (yaml) after every run, so a timeout keeps what finished.

    uv run python scripts/dataset_creation/time_butina.py \
        --dataset /dais/fs/scratch/wedigs/remedi/pretraining/pcqm4mv2 \
        --sizes 100000 300000 1000000 --coarse-cutoffs 0.6 0.65 0.7 0.8 \
        --fine-cutoff 0.4 --output butina_timing.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from remedi.training.data.molecule_clustering import (
    butina,
    cluster_size_summary,
    molecule_smiles,
    two_level_butina,
)
from remedi_data.chemistry.fingerprints import packed_morgan_fingerprints

logger = logging.getLogger("time_butina")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--coarse-cutoffs", type=float, nargs="+", required=True)
    parser.add_argument("--fine-cutoff", type=float, required=True)
    parser.add_argument("--budget-seconds", type=float, default=3 * 3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

    molecules = molecule_smiles(arguments.dataset)
    order = np.random.default_rng(arguments.seed).permutation(len(molecules))
    largest = min(max(arguments.sizes), len(molecules))
    smiles = molecules["nonisomeric_smiles"].to_numpy()[order[:largest]].tolist()
    started = time.monotonic()
    fingerprints = packed_morgan_fingerprints(smiles, workers=workers)
    report: dict = {
        "dataset": str(arguments.dataset),
        "device": torch.cuda.get_device_name(),
        "fingerprint_seconds": time.monotonic() - started,
        "fingerprints": len(fingerprints),
        "runs": [],
    }
    previous: tuple[int, float] | None = None
    for size in sorted(arguments.sizes):
        size = min(size, len(fingerprints))
        if previous is not None:
            previous_size, previous_seconds = previous
            estimate = previous_seconds * (size / previous_size) ** 2
            if estimate > arguments.budget_seconds:
                logger.warning("skipping %d: estimated %.0f s", size, estimate)
                report["runs"].append(
                    {"size": size, "skipped_estimate_seconds": estimate}
                )
                break
        sample = fingerprints[:size]
        device_sample = torch.from_numpy(sample.view(np.int32)).cuda()
        for position, cutoff in enumerate(arguments.coarse_cutoffs):
            run: dict = {"size": size, "coarse_cutoff": cutoff}
            if position == 0:
                clusters = two_level_butina(sample, cutoff, arguments.fine_cutoff)
                run |= {
                    "fine_cutoff": arguments.fine_cutoff,
                    "coarse_seconds": clusters.coarse_seconds,
                    "fine_seconds": clusters.fine_seconds,
                    **clusters.details,
                }
                previous = (size, clusters.coarse_seconds + clusters.fine_seconds)
            else:
                run_started = time.monotonic()
                coarse = butina(device_sample, cutoff)
                run |= {
                    "coarse_seconds": time.monotonic() - run_started,
                    **cluster_size_summary(coarse, "coarse"),
                }
            logger.info("%s", run)
            report["runs"].append(run)
            arguments.output.write_text(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
