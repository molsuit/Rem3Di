from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rc_params = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"""
\usepackage{helvet}     % Helvetica
\usepackage{sansmath}   % use sans serif in math mode as well
\sansmath
""",
    "font.size": 14,
}
mpl.rcParams.update(rc_params)


def _wrap(labels: Iterable[str], width: int = 16) -> list[str]:
    return [textwrap.fill(str(l), width=width) for l in labels]


def _flatten_results_any(results_any) -> pd.DataFrame:
    """
    Accept a dict {task: {model: {MAE, ...}}} or a list of such dicts.
    Return DataFrame with columns: task, model, MAE, MSE, R2 (if present).
    """
    if isinstance(results_any, dict):
        blobs = [results_any]
    elif isinstance(results_any, list):
        blobs = results_any
    else:
        raise TypeError("results must be a dict or list of dicts")
    rows = []
    for blob in blobs:
        for task, models in blob.items():
            for model, metrics in models.items():
                row = {"task": task, "model": model}
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


_NULL_REGEX = re.compile(r"(null|test\s*mean|mean\s*baseline)", re.IGNORECASE)


def _find_null_mae_for_task(df_task: pd.DataFrame) -> float | None:
    # exact hits first
    exact_keys = {"Null", "Test mean", "Mean baseline", "Null (test mean)"}
    for ek in exact_keys:
        hit = df_task.loc[df_task["model"].str.casefold() == ek.casefold()]
        if len(hit):
            return float(hit["MAE"].iloc[0])
    # fuzzy search (substring or regex match)
    for _, row in df_task.iterrows():
        if _NULL_REGEX.search(str(row["model"])):
            return float(row["MAE"])
    return None


def _auto_family_colors(
    models: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    families = {
        "REM3DI": [m for m in models if "rem3di" in m.lower()],
        "ECFP": [m for m in models if "ecfp" in m.lower()],
        "OTHER": [
            m for m in models if ("rem3di" not in m.lower() and "ecfp" not in m.lower())
        ],
    }
    colors = {}

    def assign_shades(members, cmap, lo=0.45, hi=0.85):
        if not members:
            return
        t = np.linspace(lo, hi, num=len(members))
        for name, ti in zip(members, t, strict=False):
            colors[name] = cmap(ti)

    assign_shades(families["REM3DI"], plt.cm.Blues)
    assign_shades(families["ECFP"], plt.cm.Reds)
    assign_shades(families["OTHER"], plt.cm.Greys, lo=0.35, hi=0.75)
    return colors


def plot_normalized_mae_bar_one_figure(
    results_any: Mapping[str, Any] | list[Mapping[str, Any]],
    models: list[str],
    tasks: list[str],
    *,
    title: str = "Normalized MAE vs Null (test mean)",
    save_path: str | Path | None = None,
    colors: dict[str, tuple[float, float, float, float]] | None = None,
    use_family_colors: bool = True,
) -> Path | None:
    df = _flatten_results_any(results_any)
    missing = [
        (t, m)
        for t in tasks
        for m in models
        if not ((df["task"] == t) & (df["model"] == m)).any()
    ]
    if missing:
        ex = ", ".join([f"({t}, {m})" for t, m in missing[:8]])
        if len(missing) > 8:
            ex += f", ... (+{len(missing)-8} more)"
        raise ValueError(f"Missing MAE for: {ex}")
    baselines = {}
    for t in tasks:
        df_t = df[df["task"] == t]
        base = _find_null_mae_for_task(df_t)
        if base is None:
            raise ValueError(f"No 'Null (test mean)' baseline found for task '{t}'.")
        baselines[t] = base
    norm = np.zeros((len(tasks), len(models)), dtype=float)
    for i, t in enumerate(tasks):
        base = baselines[t]
        for j, m in enumerate(models):
            mae = float(df.loc[(df["task"] == t) & (df["model"] == m), "MAE"].iloc[0])
            norm[i, j] = mae / base
    if colors is None and use_family_colors:
        colors = _auto_family_colors(models)
    elif colors is None:
        colors = {m: None for m in models}
    fig, ax = plt.subplots(figsize=(max(9.0, 1.0 * len(tasks)), 5.0), dpi=150)
    x = np.arange(len(tasks), dtype=float)
    nM = len(models)
    total_width = 0.84
    bar_w = total_width / nM
    offsets = (np.arange(nM) - (nM - 1) / 2) * bar_w
    for j, m in enumerate(models):
        ax.bar(
            x + offsets[j], norm[:, j], width=bar_w, label=m, color=colors.get(m, None)
        )

    ax.set_xticks(x, _wrap(tasks, width=14), rotation=0, ha="center")
    ax.set_ylabel("MAE / MAE(Null)")
    ax.set_title(title)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    max(1.05, float(np.nanmax(norm)) * 1.1)
    ax.set_ylim(0, 1.4)
    fig.tight_layout()
    return fig


models = ["Ridge REM3DI", "RF REM3DI", "Ridge ECFP", "RF ECFP"]
tasks_avp = ["pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)"]
tasks_adme_fang = [
    "LOG_HLM_CLint",
    "LOG_RLM_CLint",
    "LOG_MDR1-MDCK_ER",
    "LOG_HPPB",
    "LOG_RPPB",
    "LOG_SOLUBILITY",
]
tasks_av_admet = ["HLM", "KSOL", "LogD", "MDR1-MDCKII", "MLM"]

tasks_av = tasks_avp + tasks_av_admet
import yaml

results = yaml.safe_load(
    open(
        "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/test_metrics.yaml"
    )
)


fig = plot_normalized_mae_bar_one_figure(
    results_any=results, models=models, tasks=tasks_adme_fang, title="ADME Fang Dataset"
)


fig.savefig("adme_fang_pp.svg")

fig = plot_normalized_mae_bar_one_figure(
    results_any=results,
    models=models,
    tasks=tasks_av,
    title="Polaris Antiviral Dataset",
)


fig.savefig("av_pp.svg")
