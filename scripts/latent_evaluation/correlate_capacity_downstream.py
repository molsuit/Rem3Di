"""Correlate the capacity diagnostic with downstream property-prediction quality.

Reads, for every model run under an eval root (one directory per eval manifest
``output_root``):

- ``manifest.yaml``: the run's model, whose rows of the panel are used;
- ``benchmark/results.csv``: the benchmark panel;
- ``descriptor_analysis/<capacity file>``: the capacity diagnostics a
  ``descriptor_analysis`` task wrote (``kind: capacity_diagnostic``) on a fixed
  corpus, one file per variant (e.g. raw and L2-normalised).

and reports (a) the across-model Spearman correlation of every capacity metric
(effective dimension ``d_eff``, total marginal entropy ``H_tot``, per capacity
file) with the model's mean rank over the panel cells (1 = best, metric
direction aware), and (b) per panel cell, the across-model correlation of the
first capacity file's ``d_eff`` with the oriented score: is capacity predictive
of prediction quality, task by task.

Usage::

    uv run python scripts/latent_evaluation/correlate_capacity_downstream.py \\
        --eval-root /path/to/eval_runs/pcqm_ablation_novicreg \\
        --capacity-files capacity_diagnostic.yaml capacity_diagnostic_l2.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import spearmanr

from remedi_data.bundle import EvalMetric
from remedi_data.logging_setup import setup_logging
from remedi_data.runs.task_runner import MANIFEST_FILENAME
from remedi_data.yaml_io import read_model, read_yaml, write_model
from remedi_eval.benchmark.metrics import higher_is_better
from remedi_eval.framework import EvalManifest

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("benchmark") / "results.csv"
ANALYSIS_DIRECTORY = Path("descriptor_analysis")
#: Columns that identify one panel cell across models.
CELL_COLUMNS = ["dataset_id", "target_col", "learner_name", "metric_name"]


class CapacityCorrelationReport(BaseModel):
    """Serializable summary of the capacity-vs-downstream analysis."""

    models: list[str]
    capacity_files: list[str]
    #: capacity metric -> Spearman correlation with the mean panel rank
    #: (negative: more capacity, better rank).
    spearman_with_mean_rank: dict[str, float]
    #: Summaries of the per-cell ``d_eff`` correlations, by subgroup.
    per_cell_effective_dimension: dict[str, dict[str, float]]
    cell_count: int


def model_runs(eval_root: Path, capacity_files: list[str]) -> dict[str, Path]:
    """The model name of every run under ``eval_root`` that has a panel and
    every capacity file, mapped to its directory.

    Raises:
        FileNotFoundError: if no run has them all.
    """
    runs = {}
    for directory in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        needed = [directory / MANIFEST_FILENAME, directory / RESULTS_PATH] + [
            directory / ANALYSIS_DIRECTORY / name for name in capacity_files
        ]
        if all(path.is_file() for path in needed):
            manifest = read_model(directory / MANIFEST_FILENAME, EvalManifest)
            runs[manifest.model.name] = directory
    if not runs:
        raise FileNotFoundError(
            f"no run under {eval_root} has {RESULTS_PATH} and "
            f"{ANALYSIS_DIRECTORY}/{{{', '.join(capacity_files)}}}"
        )
    return runs


def capacity_table(runs: dict[str, Path], capacity_files: list[str]) -> pd.DataFrame:
    """``d_eff`` and ``H_tot`` of every capacity file, one row per model."""
    rows = []
    for model, directory in runs.items():
        row: dict[str, float | str] = {"model": model}
        for name in capacity_files:
            report = read_yaml(directory / ANALYSIS_DIRECTORY / name)
            stem = Path(name).stem
            row[f"{stem}:d_eff"] = float(report["d_eff"])
            row[f"{stem}:H_tot"] = float(report["H_tot"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def panel_table(runs: dict[str, Path]) -> pd.DataFrame:
    """Every model's own rows of its panel, with an oriented score and the
    model's rank within each cell."""
    frames = []
    for model, directory in runs.items():
        rows = pd.read_csv(directory / RESULTS_PATH)
        rows = rows[rows["descriptor_name"] == model].assign(model=model)
        frames.append(rows)
    panel = pd.concat(frames, ignore_index=True)
    panel["target_col"] = panel["target_col"].fillna("")
    sign = panel["metric_name"].map(
        lambda name: 1.0 if higher_is_better(EvalMetric(name)) else -1.0
    )
    panel["oriented"] = sign * panel["metric_value"]
    panel["rank"] = panel.groupby(CELL_COLUMNS)["oriented"].rank(
        ascending=False, method="average"
    )
    return panel


def _summarize(values: pd.Series) -> dict[str, float]:
    values = values.dropna()
    return {
        "median": float(values.median()),
        "mean": float(values.mean()),
        "fraction_positive": float((values > 0).mean()),
        "count": int(values.size),
    }


def run(
    eval_root: Path, capacity_files: list[str]
) -> tuple[CapacityCorrelationReport, pd.DataFrame]:
    runs = model_runs(eval_root, capacity_files)
    capacity = capacity_table(runs, capacity_files)
    panel = panel_table(runs)
    mean_rank = panel.groupby("model")["rank"].mean()

    with_mean_rank = {
        metric: float(
            spearmanr(capacity.loc[mean_rank.index, metric], mean_rank).statistic
        )
        for metric in capacity.columns
    }

    effective_dimension = f"{Path(capacity_files[0]).stem}:d_eff"
    cells = []
    for key, rows in panel.groupby(CELL_COLUMNS):
        rows = rows.set_index("model")
        if rows.index.nunique() < len(runs) or rows["oriented"].nunique() < 3:
            continue
        result = spearmanr(
            capacity.loc[rows.index, effective_dimension].to_numpy(dtype=float),
            rows["oriented"].to_numpy(dtype=float),
        )
        cells.append(
            dict(zip(CELL_COLUMNS, key, strict=True))
            | {
                "learner_kind": rows["learner_kind"].iloc[0],
                "spearman": result.statistic,
                "p_value": result.pvalue,
            }
        )
    per_cell = pd.DataFrame(cells)
    if per_cell.empty:
        raise ValueError(
            "no panel cell is scored for every model with at least three distinct values"
        )

    summaries = {"all": _summarize(per_cell["spearman"])}
    for learner_kind, rows in per_cell.groupby("learner_kind"):
        summaries[f"learner_kind={learner_kind}"] = _summarize(rows["spearman"])
    significant = per_cell[per_cell["p_value"] < 0.05]
    summaries["significant_p05"] = {
        "fraction_of_cells": float((per_cell["p_value"] < 0.05).mean()),
        "fraction_positive": float((significant["spearman"] > 0).mean())
        if len(significant)
        else float(np.nan),
    }
    report = CapacityCorrelationReport(
        models=list(runs),
        capacity_files=capacity_files,
        spearman_with_mean_rank=with_mean_rank,
        per_cell_effective_dimension=summaries,
        cell_count=len(per_cell),
    )
    return report, per_cell


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument(
        "--capacity-files",
        nargs="+",
        default=["capacity_diagnostic.yaml"],
        help="Capacity reports under each run's descriptor_analysis/; the first "
        "one's d_eff is the per-cell correlate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write capacity_correlation.yaml + per_cell_capacity.csv "
        "(default: <eval-root>/capacity_correlation).",
    )
    args = parser.parse_args()

    report, per_cell = run(args.eval_root, args.capacity_files)
    out = args.output_dir or (args.eval_root / "capacity_correlation")
    out.mkdir(parents=True, exist_ok=True)
    write_model(out / "capacity_correlation.yaml", report)
    per_cell.to_csv(out / "per_cell_capacity.csv", index=False)

    logger.info("Spearman of capacity with the mean panel rank:")
    for metric, value in report.spearman_with_mean_rank.items():
        logger.info("  %-40s %+.2f", metric, value)
    logger.info("per-cell d_eff correlation:")
    for group, summary in report.per_cell_effective_dimension.items():
        logger.info("  %s: %s", group, summary)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
