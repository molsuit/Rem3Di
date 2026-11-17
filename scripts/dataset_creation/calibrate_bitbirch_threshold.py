"""Report mean iSIM, std, and threshold candidates for a DatasetComparison
configuration without running BitBIRCH.

bblean's recommended threshold is ``mean_isim + factor * std`` (factor=3 by
default). This script lets you pick `bitbirch.auto_threshold_factor` (or set
`bitbirch.threshold` directly) before paying for a full BitBIRCH fit. It
fingerprints every dataset in the config and reports both per-dataset and
union iSIM/std so you can see whether the eval sets sit in a tighter or
looser neighbourhood than the pretrain pool.

Usage::

    uv run python scripts/dataset_creation/calibrate_bitbirch_threshold.py \\
        --config configs/dataset_creation/dataset_comparison_dev.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from remedi_data.logging_setup import setup_logging
from remedi_data.yaml_io import read_model
from remedi_eval.dataset_analysis.comparison import DatasetComparison
from remedi_eval.dataset_analysis.config import (
    DatasetComparisonConfig,
)

DEFAULT_FACTORS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

log = logging.getLogger(__name__)


def _isim_pair(fps_packed: np.ndarray, n_features: int) -> tuple[float, float]:
    import bblean.similarity as sim

    mean = float(sim.jt_isim_packed(fps_packed, n_features=n_features))
    std = float(
        sim.estimate_jt_std(fps_packed, input_is_packed=True, n_features=n_features)
    )
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=list(DEFAULT_FACTORS),
        help="Factors to evaluate in the threshold = mean + factor*std heuristic.",
    )
    args = parser.parse_args()

    setup_logging()

    config = read_model(args.config, DatasetComparisonConfig)
    n_features = config.bitbirch.fingerprint_size
    comparison = DatasetComparison(config)
    comparison.load_datasets()
    union = comparison.union_fingerprints()

    print("\nPer-dataset iSIM:")
    print("-" * 72)
    print(f"  {'name':<24}{'role':<10}{'n_fp':>8}  {'mean iSIM':>10}  {'std':>10}")
    for ld in comparison.loaded_datasets:
        if ld.fps_packed.shape[0] < 2:
            print(
                f"  {ld.entry.name:<24}{ld.entry.role:<10}"
                f"{ld.fps_packed.shape[0]:>8}  (too few molecules)"
            )
            continue
        mean, std = _isim_pair(ld.fps_packed, n_features)
        print(
            f"  {ld.entry.name:<24}{ld.entry.role:<10}"
            f"{ld.fps_packed.shape[0]:>8}  {mean:>10.4f}  {std:>10.4f}"
        )

    print("\nUnion iSIM (used by `auto_threshold`):")
    print("-" * 72)
    union_mean, union_std = _isim_pair(union.fps_packed, n_features)
    print(
        f"  n_fp={union.fps_packed.shape[0]}  mean iSIM={union_mean:.4f}"
        f"  std={union_std:.4f}"
    )

    print("\nThreshold candidates (mean + factor * std):")
    print("-" * 72)
    print(f"  {'factor':>8}  {'threshold':>10}")
    for f in args.factors:
        thr = float(np.clip(union_mean + f * union_std, 0.0, 1.0))
        print(f"  {f:>8.2f}  {thr:>10.4f}")
    print(
        "\nTo use the auto-calibrated threshold in the comparison, set:\n"
        "  bitbirch:\n"
        "    auto_threshold: true\n"
        "    auto_threshold_factor: <factor>\n"
        "in the yaml. Otherwise pick a value from the table above and set\n"
        "  bitbirch.threshold: <value>\n"
    )


if __name__ == "__main__":
    main()
