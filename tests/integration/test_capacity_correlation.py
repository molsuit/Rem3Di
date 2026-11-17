"""``scripts/latent_evaluation/correlate_capacity_downstream.py`` on synthetic runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from remedi_data.runs.task_runner import MANIFEST_FILENAME
from remedi_data.yaml_io import write_yaml

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "latent_evaluation"
    / "correlate_capacity_downstream.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(
    root: Path, model: str, effective_dimension: float, scores: list[float]
) -> None:
    directory = root / model
    (directory / "benchmark").mkdir(parents=True)
    (directory / "descriptor_analysis").mkdir()
    write_yaml(
        directory / MANIFEST_FILENAME,
        {
            "model": {"kind": "ecfp", "name": model, "fingerprint_size": 256},
            "output_root": str(directory),
            "tasks": [
                {
                    "kind": "descriptor_analysis",
                    "dataset_path": "corpus",
                    "tasks": [{"kind": "capacity_diagnostic"}],
                }
            ],
        },
    )
    write_yaml(
        directory / "descriptor_analysis" / "capacity_diagnostic.yaml",
        {"d_eff": effective_dimension, "H_tot": 2 * effective_dimension},
    )
    rows = [
        {
            "dataset_id": f"set{index}",
            "descriptor_name": descriptor,
            "learner_name": "linear",
            "learner_kind": "linear",
            "target_col": "y",
            "metric_name": "RMSE",
            "metric_value": score if descriptor == model else 0.0,
        }
        for index, score in enumerate(scores)
        for descriptor in (model, "ecfp_baseline")
    ]
    pd.DataFrame(rows).to_csv(directory / "benchmark" / "results.csv", index=False)


def test_capacity_correlates_with_the_models_own_panel_rows(tmp_path: Path) -> None:
    """More capacity, lower RMSE in every cell: the per-cell correlation of
    ``d_eff`` with the oriented score is +1, the one with the mean rank -1, and
    the baseline rows of each run are ignored."""
    module = _load_script()
    _write_run(tmp_path, "small", 2.0, [3.0, 3.0])
    _write_run(tmp_path, "medium", 4.0, [2.0, 2.5])
    _write_run(tmp_path, "large", 8.0, [1.0, 1.0])

    report, per_cell = module.run(tmp_path, ["capacity_diagnostic.yaml"])

    assert sorted(report.models) == ["large", "medium", "small"]
    assert report.cell_count == 2
    assert per_cell["spearman"].tolist() == pytest.approx([1.0, 1.0])
    assert report.spearman_with_mean_rank["capacity_diagnostic:d_eff"] == pytest.approx(
        -1.0
    )


def test_a_root_without_complete_runs_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "half_run" / "benchmark").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no run under"):
        _load_script().run(tmp_path, ["capacity_diagnostic.yaml"])
