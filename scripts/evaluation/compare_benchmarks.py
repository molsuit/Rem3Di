"""Aggregate benchmark results across models + baselines into leaderboards.

Reads every ``<eval_root>/<model>/benchmark/results.csv`` (REM3DI models *and*
baselines such as ``ecfp_2048``), concatenates them into one tidy frame, and
emits leaderboards. Because metrics mix directions and scales (RMSE/MAE lower is
better; AUROC/AUPRC/Spearman higher), aggregation is on two comparable axes:

* **mean rank** — within each ``(dataset, target, metric)`` cell, every
  ``(model, learner)`` competitor is ranked direction-aware (1 = best); averaged
  per competitor over all cells.
* **rel. improvement vs null** — per cell, relative to the null baseline's value,
  signed so positive always means "better than null" regardless of metric
  direction. Undefined (NaN) where the null value is ~0 (e.g. Spearman).

Outputs under ``--output-dir`` (default ``<eval_root>/comparison``):
``long.csv``, ``per_task.csv`` (wide), ``leaderboard.csv``,
``leaderboard_by_tasktype.csv``.

Usage::

    uv run python scripts/evaluation/compare_benchmarks.py \\
        --eval-root /path/to/evaluation_results/pcqm_ablation_novicreg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOWER_IS_BETTER = {"RMSE", "MAE", "MSE"}
REGRESSION_METRICS = {"RMSE", "MAE", "MSE", "Spearman"}
_EPS = 1e-9


def load_long(eval_root: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(eval_root.glob("*/benchmark/results.csv")):
        model = csv.parent.parent.name
        df = pd.read_csv(csv)
        df["model"] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No <model>/benchmark/results.csv under {eval_root}")
    long = pd.concat(frames, ignore_index=True)
    long["target_col"] = long["target_col"].fillna("_")
    long["ttype"] = np.where(
        long.metric_name.isin(REGRESSION_METRICS), "regression", "classification"
    )
    long["competitor"] = long["model"] + " / " + long["learner_kind"]
    long["cell"] = (
        long.dataset_id + "|" + long.target_col + "|" + long.metric_name
    )
    return long


def add_rank(long: pd.DataFrame) -> pd.DataFrame:
    def _rank(g: pd.DataFrame) -> pd.Series:
        ascending = g.metric_name.iloc[0] not in LOWER_IS_BETTER  # higher=better -> desc
        return g.metric_value.rank(ascending=not ascending, method="average")

    long = long.copy()
    long["rank"] = long.groupby("cell", group_keys=False).apply(_rank)
    return long


def add_rel_improvement(
    long: pd.DataFrame, null_model: str, null_learner: str
) -> pd.DataFrame:
    null = (
        long[(long.model == null_model) & (long.learner_kind == null_learner)]
        .set_index("cell")["metric_value"]
        .rename("null_value")
    )
    if null.empty:
        logger.warning(
            "no null rows (model=%s learner=%s); rel_improvement will be NaN",
            null_model, null_learner,
        )
    long = long.merge(null, left_on="cell", right_index=True, how="left")

    lower = long.metric_name.isin(LOWER_IS_BETTER)
    nv = long.null_value
    safe = nv.abs() > _EPS
    rel = np.where(
        lower,
        (nv - long.metric_value) / nv,          # lower better: shrink error
        (long.metric_value - nv) / nv,          # higher better: grow score
    )
    long["rel_improvement_vs_null"] = np.where(safe, rel, np.nan)
    return long


def leaderboard(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = long.groupby(by)
    out = pd.DataFrame(
        {
            "n_cells": g.size(),
            "mean_rank": g["rank"].mean(),
            "median_rank": g["rank"].median(),
            "mean_rel_impr_vs_null": g["rel_improvement_vs_null"].mean(),
        }
    ).reset_index()
    return out.sort_values("mean_rank").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--null-model", type=str, default="ecfp_2048")
    parser.add_argument("--null-learner", type=str, default="null_baseline")
    args = parser.parse_args()

    out = args.output_dir or (args.eval_root / "comparison")
    out.mkdir(parents=True, exist_ok=True)

    long = load_long(args.eval_root)
    long = add_rank(long)
    long = add_rel_improvement(long, args.null_model, args.null_learner)

    long.to_csv(out / "long.csv", index=False)
    long.pivot_table(
        index=["dataset_id", "target_col", "metric_name"],
        columns="competitor",
        values="metric_value",
    ).to_csv(out / "per_task.csv")

    lb = leaderboard(long, ["model", "learner_kind"])
    lb.to_csv(out / "leaderboard.csv", index=False)
    leaderboard(long, ["model", "learner_kind", "ttype"]).to_csv(
        out / "leaderboard_by_tasktype.csv", index=False
    )

    logger.info(
        "models=%d, competitors=%d, task-cells=%d -> %s",
        long.model.nunique(), long.competitor.nunique(), long.cell.nunique(), out,
    )
    logger.info("top of leaderboard (by mean rank):\n%s", lb.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
