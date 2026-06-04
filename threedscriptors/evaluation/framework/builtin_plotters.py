"""Built-in plotters for framework data artifacts (decoupled from tasks).

Importing this module registers the plotters; ``replot.py`` imports it for the
side effect. Plotters consume a *loaded* data artifact (e.g. the benchmark
``results.csv`` DataFrame) and return :class:`FigureResult` objects, so figures
can be regenerated offline from saved artifacts without re-running the eval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from threedscriptors.evaluation.framework.plotting import register_plotter
from threedscriptors.evaluation.results import FigureResult

# Lower-is-better metrics, so bar charts can be oriented / read correctly.
_LOWER_IS_BETTER = {"RMSE", "MAE", "MSE"}


@register_plotter("benchmark_results")
def plot_benchmark_results(df: pd.DataFrame, out_dir: Path) -> list[FigureResult]:
    """One horizontal grouped bar chart per metric: value by dataset x learner.

    Multi-target datasets are averaged over their target columns so each
    (dataset, learner) is one bar.
    """
    del out_dir  # paths are assigned via FigureResult.file_name
    figures: list[FigureResult] = []
    for metric, df_m in df.groupby("metric_name"):
        agg = (
            df_m.groupby(["dataset_id", "learner_kind"])["metric_value"]
            .mean()
            .unstack("learner_kind")
            .sort_index()
        )
        n = len(agg)
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * n + 1.5)))
        agg.plot.barh(ax=ax)
        direction = "lower is better" if metric in _LOWER_IS_BETTER else "higher is better"
        ax.set_xlabel(f"{metric}  ({direction})")
        ax.set_ylabel("")
        ax.set_title(f"Benchmark: {metric}")
        ax.legend(title="learner", fontsize="small")
        fig.tight_layout()
        figures.append(
            FigureResult(file_name=Path(f"benchmark_{metric}.png"), figure=fig)
        )
    return figures
