"""Built-in plotters for framework data artifacts (decoupled from tasks).

Importing this module registers the plotters. Plotters consume a *loaded* data
artifact (e.g. the benchmark ``results.csv`` DataFrame) and return
:class:`FigureResult` objects, so figures can be regenerated offline from saved
artifacts without re-running the eval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from remedi_data.bundle import EvalMetric
from remedi_eval.benchmark.metrics import higher_is_better
from remedi_eval.framework.plotting import register_plotter
from remedi_eval.results import FigureResult


@register_plotter("benchmark_results")
def plot_benchmark_results(df: pd.DataFrame, out_dir: Path) -> list[FigureResult]:
    """One horizontal grouped bar chart per metric: value by dataset, one bar
    per (descriptor, learner name).

    The model and each baseline descriptor get their own bars, as do two
    learners of one kind. Multi-target datasets are averaged over their target
    columns so each (dataset, descriptor, learner) is one bar.
    """
    del out_dir  # paths are assigned via FigureResult.file_name
    figures: list[FigureResult] = []
    for metric, df_m in df.groupby("metric_name"):
        series = df_m.assign(
            series=df_m["descriptor_name"] + " / " + df_m["learner_name"]
        )
        agg = (
            series.groupby(["dataset_id", "series"])["metric_value"]
            .mean()
            .unstack("series")
            .sort_index()
        )
        n = len(agg)
        fig, ax = plt.subplots(
            figsize=(8, max(3.0, 0.4 * n * max(1, agg.shape[1] / 2) + 1.5))
        )
        agg.plot.barh(ax=ax)
        direction = (
            "higher is better"
            if higher_is_better(EvalMetric(str(metric)))
            else "lower is better"
        )
        ax.set_xlabel(f"{metric}  ({direction})")
        ax.set_ylabel("")
        ax.set_title(f"Benchmark: {metric}")
        ax.legend(title="descriptor / learner", fontsize="small")
        fig.tight_layout()
        figures.append(
            FigureResult(file_name=Path(f"benchmark_{metric}.png"), figure=fig)
        )
    return figures


@register_plotter("confusion_matrix")
def plot_confusion_matrix(
    arrays: dict[str, np.ndarray], out_dir: Path
) -> list[FigureResult]:
    """Row-normalised confusion-matrix heatmap (recall per true class).

    Consumes a loaded ``confusion_matrix.npz`` — ``confusion_matrix`` plus the
    display ``labels`` — which every multiclass benchmark cell writes. Each
    tile is annotated with the raw count over its row-normalised share, so the
    rare classes stay readable next to the dominant ones.
    """
    del out_dir  # paths are assigned via FigureResult.file_name
    counts = np.asarray(arrays["confusion_matrix"])
    names = [str(label) for label in np.asarray(arrays["labels"])]

    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )

    n = len(names)
    fig, ax = plt.subplots(figsize=(1.4 * n + 1.5, 1.4 * n + 1.0))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for row in range(n):
        for column in range(n):
            ax.text(
                column,
                row,
                f"{counts[row, column]}\n{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return [FigureResult(file_name=Path("confusion_matrix.png"), figure=fig)]
