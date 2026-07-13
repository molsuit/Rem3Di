"""Correlate the capacity diagnostic with downstream property-prediction quality.

Reproduces the analysis written up in ``docs/pcqm_novicreg_eval_results.md``.

For every model under an eval root (one ``<model>/benchmark/results.csv`` +
``<model>/retrieval/tanimoto_similarity.yaml`` each, plus a shared
``descriptor_cache/pcqm100k__<model>.npz``) it:

1. Recomputes the latent capacity diagnostic (effective dimension ``d_eff``,
   total marginal entropy ``H_tot``) on the *cached* PCQM100k embeddings — no
   GPU / re-encoding needed.
2. Aggregates the benchmark panel into a per-model mean rank (1 = best across
   models, metric-direction aware) and reads the retrieval Spearman.
3. Reports (a) aggregate across-model correlations of each capacity metric vs
   the two downstream summaries, and (b) the per-task distribution of the
   across-model correlation between ``d_eff`` and oriented task performance —
   i.e. is capacity predictive of prediction quality, task by task.

Usage::

    uv run python scripts/evaluation/correlate_capacity_downstream.py \\
        --eval-root /path/to/evaluation_results/pcqm_ablation_novicreg
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
import yaml
from pydantic import BaseModel
from scipy.stats import spearmanr

from remedi.evaluation.descriptor_analysis.capacity_diagnostic import (
    run_latent_space_capacity_diagnostic,
)

logger = logging.getLogger(__name__)

# Benchmark metrics where lower is better (everything else is higher-is-better).
LOWER_IS_BETTER = {"RMSE", "MAE", "MSE"}
# Regression-style metrics (for the regression/classification task-type split).
REGRESSION_METRICS = {"RMSE", "MAE", "MSE", "Spearman"}
CAPACITY_METRICS = ("deff", "deff_l2", "H_tot")


class CapacityCorrelationReport(BaseModel):
    """Serializable summary of the capacity-vs-downstream analysis."""

    models: list[str]
    retrieval_dataset_id: str
    # aggregate: capacity metric -> {"vs_bench_rank": rho, "vs_retr_spearman": rho}
    aggregate_spearman: dict[str, dict[str, float]]
    # per-task d_eff correlation distribution summaries, keyed by subgroup
    per_task_deff: dict[str, dict[str, float]]
    n_tasks: int
    n_tasks_regression: int
    n_tasks_classification: int


def _capacity_per_model(
    cache_dir: Path, dataset_id: str, models: list[str]
) -> pd.DataFrame:
    rows = []
    for m in models:
        X = np.load(cache_dir / f"{dataset_id}__{m}.npz")["X"]
        rows.append(
            {
                "model": m,
                "D": int(X.shape[1]),
                "deff": run_latent_space_capacity_diagnostic(X).d_eff,
                "deff_l2": run_latent_space_capacity_diagnostic(
                    X, l2_normalize=True
                ).d_eff,
                "H_tot": run_latent_space_capacity_diagnostic(X).H_tot,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def _benchmark_long(eval_root: Path, models: list[str]) -> pd.DataFrame:
    frames = []
    for m in models:
        df = pd.read_csv(eval_root / m / "benchmark" / "results.csv")
        df["model"] = m
        frames.append(df)
    bdf = pd.concat(frames, ignore_index=True)
    bdf["oriented"] = np.where(
        bdf.metric_name.isin(LOWER_IS_BETTER), -bdf.metric_value, bdf.metric_value
    )
    bdf["ttype"] = np.where(
        bdf.metric_name.isin(REGRESSION_METRICS), "regression", "classification"
    )
    bdf["cell"] = (
        bdf.dataset_id + "|" + bdf.target_col.fillna("_") + "|" + bdf.learner_kind
    )
    bdf["rank"] = bdf.groupby("cell", group_keys=False).apply(
        lambda g: g.metric_value.rank(
            ascending=g.metric_name.iloc[0] in LOWER_IS_BETTER, method="average"
        )
    )
    return bdf


def _summarize(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    return {
        "median": float(s.median()),
        "mean": float(s.mean()),
        "frac_positive": float((s > 0).mean()),
        "n": int(s.size),
    }


def run(
    eval_root: Path, dataset_id: str
) -> tuple[CapacityCorrelationReport, pd.DataFrame]:
    cache_dir = eval_root / "descriptor_cache"
    models = sorted(
        Path(p).name.removeprefix(f"{dataset_id}__").removesuffix(".npz")
        for p in glob.glob(str(cache_dir / f"{dataset_id}__*.npz"))
    )
    if not models:
        raise FileNotFoundError(f"No cached {dataset_id} embeddings under {cache_dir}")

    cap = _capacity_per_model(cache_dir, dataset_id, models)
    bdf = _benchmark_long(eval_root, models)
    mean_rank = bdf.groupby("model")["rank"].mean()
    retr = {
        m: yaml.safe_load(
            (eval_root / m / "retrieval" / "tanimoto_similarity.yaml").read_text()
        )["spearman_emb_vs_tanimoto"]
        for m in models
    }
    retr_s = pd.Series(retr)

    aggregate: dict[str, dict[str, float]] = {}
    for c in CAPACITY_METRICS:
        rb = spearmanr(cap.loc[mean_rank.index, c], mean_rank).statistic
        rr = spearmanr(cap.loc[retr_s.index, c], retr_s).statistic
        aggregate[c] = {"vs_bench_rank": float(rb), "vs_retr_spearman": float(rr)}

    # Per-task: across-model correlation of each capacity metric vs oriented perf.
    recs = []
    grouped = bdf.groupby(["dataset_id", "target_col", "learner_kind"], dropna=False)
    for group_key, g in grouped:
        ds, tc, lk = group_key  # ty: ignore[not-iterable]  # multi-key groupby
        g = g.set_index("model")
        if g.index.nunique() < len(models) or g["oriented"].nunique() < 3:
            continue
        rec = {
            "dataset_id": ds,
            "target_col": tc,
            "learner": lk,
            "ttype": g.ttype.iloc[0],
        }
        for c in CAPACITY_METRICS:
            res = spearmanr(cap.loc[g.index, c].astype(float), g["oriented"].values)
            rec[f"rho_{c}"] = res.statistic
            rec[f"p_{c}"] = res.pvalue
        recs.append(rec)
    pt = pd.DataFrame(recs)

    per_task = {
        "all": _summarize(pt.rho_deff),
        "regression": _summarize(pt.loc[pt.ttype == "regression", "rho_deff"]),
        "classification": _summarize(pt.loc[pt.ttype == "classification", "rho_deff"]),
        "linear": _summarize(pt.loc[pt.learner == "linear", "rho_deff"]),
        "mlp": _summarize(pt.loc[pt.learner == "mlp", "rho_deff"]),
    }
    sig = pt[pt.p_deff < 0.05]
    per_task["significant_p05"] = {
        "frac_of_tasks": float((pt.p_deff < 0.05).mean()),
        "frac_positive_among_significant": float((sig.rho_deff > 0).mean())
        if len(sig)
        else float("nan"),
    }

    report = CapacityCorrelationReport(
        models=models,
        retrieval_dataset_id=dataset_id,
        aggregate_spearman=aggregate,
        per_task_deff=per_task,
        n_tasks=len(pt),
        n_tasks_regression=int((pt.ttype == "regression").sum()),
        n_tasks_classification=int((pt.ttype == "classification").sum()),
    )
    return report, pt


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/path/to/evaluation_results/pcqm_ablation_novicreg"),
    )
    parser.add_argument("--dataset-id", type=str, default="pcqm100k")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write capacity_correlation.yaml + per_task_capacity.csv "
        "(default: <eval-root>/capacity_correlation).",
    )
    args = parser.parse_args()

    report, pt = run(args.eval_root, args.dataset_id)
    out = args.output_dir or (args.eval_root / "capacity_correlation")
    out.mkdir(parents=True, exist_ok=True)
    pyd_yaml.to_yaml_file(out / "capacity_correlation.yaml", report)
    pt.to_csv(out / "per_task_capacity.csv", index=False)

    logger.info("aggregate Spearman (capacity vs downstream):")
    for c, d in report.aggregate_spearman.items():
        logger.info(
            "  %-8s vs bench_rank %+.2f   vs retr_spearman %+.2f",
            c,
            d["vs_bench_rank"],
            d["vs_retr_spearman"],
        )
    logger.info("per-task d_eff correlation (median, frac positive):")
    for k, d in report.per_task_deff.items():
        logger.info("  %s", f"{k}: {d}")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
