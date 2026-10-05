"""End-to-end ``run_eval`` over datasets written by the real ``write_dataset``.

Pins: one row per scored target, the cell identity (``split_column`` / ``seed``)
reaches ``results.csv``, a non-default split column is read from the table, a
corpus is not scored, and one failing benchmark does not sink the panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from remedi.data_handling.bundle import EvalMetric, LabelColumn
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.descriptors import EcfpConfig
from remedi.evaluation.benchmark.learners import LinearLearnerConfig
from remedi.evaluation.benchmark.runner import EvalConfig, run_eval

from .helpers.bundle_fixtures import (
    ACHIRAL_TEN_SMILES,
    TEN_ROW_SPLIT,
    write_small_dataset,
)

N_ROWS = len(ACHIRAL_TEN_SMILES)
#: A second, deliberately differently *sized* partition that exists only in the
#: table, never in the zarr's ``tasks/split`` array: two train, two valid, six
#: test, so the fold counts alone say which column was read.
ALTERNATIVE_SPLIT = ["train"] * 2 + ["valid"] * 2 + ["test"] * 6
#: Alternating labels, so the two test rows carry both classes (AUROC needs it).
ALTERNATING = (np.arange(N_ROWS) % 2).astype(float)


def _labels(task_type: TaskType, *names: str) -> list[LabelColumn]:
    return [LabelColumn(name=name, task_type=task_type) for name in names]


def build_zarr(
    tmp_path: Path,
    dataset_id: str = "toy_reg",
    *,
    labels: list[LabelColumn] | None = None,
    metrics: tuple[EvalMetric, ...] | None = (EvalMetric.rmse,),
    targets: np.ndarray | None = None,
    splits: dict[str, list[str]] | None = None,
) -> Path:
    """One dataset under ``tmp_path/datasets``; a single regression label by default."""
    return write_small_dataset(
        tmp_path / "datasets",
        dataset_id=dataset_id,
        labels=labels or _labels(TaskType.regression, "y"),
        metrics=None if metrics is None else list(metrics),
        targets=np.linspace(0.0, 1.0, N_ROWS) if targets is None else targets,
        splits=splits,
    )


def eval_config(tmp_path: Path, **overrides: Any) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "eval_root": tmp_path / "datasets",
            "output_dir": tmp_path / "eval_out",
            "descriptors": [EcfpConfig(name="ecfp_128", length=128)],
            "learners": [LinearLearnerConfig(ridge_alpha=1.0)],
            **overrides,
        }
    )


def _with_nan_holes(targets: np.ndarray) -> np.ndarray:
    """Holes in the train fold of column 1: NaN labels are dropped, not fatal."""
    targets = targets.copy()
    targets[[0, 3], 1] = np.nan
    return targets


# ------------------------------------------------------------------ the cells


def test_run_eval_cross_product_and_cell_identity(tmp_path: Path) -> None:
    build_zarr(tmp_path)

    rows = run_eval(
        eval_config(
            tmp_path,
            seed=7,
            descriptors=[
                EcfpConfig(name="ecfp_128", length=128),
                EcfpConfig(name="ecfp_512", length=512),
            ],
            learners=[
                LinearLearnerConfig(ridge_alpha=0.1),
                LinearLearnerConfig(ridge_alpha=10.0),
            ],
        )
    )

    # 1 benchmark x 2 descriptors x 2 learners = 4 rows.
    assert len(rows) == 4
    assert {row.descriptor_name for row in rows} == {"ecfp_128", "ecfp_512"}
    for row in rows:
        assert (row.dataset_id, row.learner_kind) == ("toy_reg", "linear")
        assert (row.n_train, row.n_val, row.n_test) == (6, 2, 2)
        # The spec's own default_split, and the config's seed.
        assert (row.split_column, row.seed) == ("split", 7)

    csv = pd.read_csv(tmp_path / "eval_out" / "results.csv")
    assert len(csv) == 4
    assert "source" not in csv.columns
    assert set(csv["split_column"]) == {"split"}
    assert (tmp_path / "eval_out" / "results.yaml").exists()


@pytest.mark.parametrize(
    ("labels", "metric", "targets", "expected_target_columns"),
    [
        (_labels(TaskType.classification, "y"), EvalMetric.auroc, ALTERNATING, {"y"}),
        (
            _labels(TaskType.regression, "y0", "y1", "y2"),
            EvalMetric.mae,
            _with_nan_holes(np.random.default_rng(0).normal(size=(N_ROWS, 3))),
            {"y0", "y1", "y2"},
        ),
        # Multilabel classification: a single row over every column.
        (
            _labels(TaskType.classification, "a", "b"),
            EvalMetric.macro_auroc,
            _with_nan_holes(np.stack([ALTERNATING, 1.0 - ALTERNATING], axis=1)),
            {None},
        ),
    ],
    ids=["binary", "multitarget_regression", "multilabel"],
)
def test_run_eval_scores_one_row_per_target(
    tmp_path: Path,
    labels: list[LabelColumn],
    metric: EvalMetric,
    targets: np.ndarray | None,
    expected_target_columns: set[str | None],
) -> None:
    build_zarr(tmp_path, labels=labels, metrics=(metric,), targets=targets)

    rows = run_eval(eval_config(tmp_path))

    assert len(rows) == len(expected_target_columns)
    assert {row.target_col for row in rows} == expected_target_columns
    assert all(row.metric_name == metric.value for row in rows)
    assert all(np.isfinite(row.metric_value) for row in rows)
    assert not (tmp_path / "eval_out" / "failures.yaml").exists()


# ------------------------------------------------------------- split columns


def test_split_column_selection(tmp_path: Path) -> None:
    """The zarr stores only the default split; the rest come from the table, and
    an unknown column fails that benchmark into ``failures.yaml``."""
    build_zarr(
        tmp_path,
        splits={"split": list(TEN_ROW_SPLIT), "split__seed1": ALTERNATIVE_SPLIT},
    )

    (row,) = run_eval(eval_config(tmp_path, split_column="split__seed1"))
    assert (row.n_train, row.n_val, row.n_test) == (2, 2, 6)
    assert row.split_column == "split__seed1"

    assert run_eval(eval_config(tmp_path, split_column="split__nope")) == []
    assert "split__nope" in (tmp_path / "eval_out" / "failures.yaml").read_text()


# ---------------------------------------------------------- fault tolerance


def test_run_eval_keeps_going_when_one_benchmark_fails(tmp_path: Path) -> None:
    """A mixed-task-type benchmark is recorded in failures.yaml and skipped."""
    build_zarr(tmp_path, "toy_ok")
    build_zarr(
        tmp_path,
        "toy_mixed",
        labels=[
            LabelColumn(name="r", task_type=TaskType.regression),
            LabelColumn(name="c", task_type=TaskType.classification),
        ],
        targets=np.stack([np.linspace(0.0, 1.0, N_ROWS), ALTERNATING], axis=1),
    )

    rows = run_eval(eval_config(tmp_path))

    assert {row.dataset_id for row in rows} == {"toy_ok"}
    csv = pd.read_csv(tmp_path / "eval_out" / "results.csv")
    assert set(csv["dataset_id"]) == {"toy_ok"}
    assert "toy_mixed" in (tmp_path / "eval_out" / "failures.yaml").read_text()


def test_discovery_scores_only_benchmarks(tmp_path: Path) -> None:
    """A corpus (no evaluation block) and the build run's own directories and
    ``status.yaml`` under ``zarr_root`` are skipped, not failures."""
    build_zarr(tmp_path)
    build_zarr(tmp_path, "toy_corpus", metrics=None)
    (tmp_path / "datasets" / "build").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "verify").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "status.yaml").write_text("n_tasks: 1\n")

    rows = run_eval(eval_config(tmp_path))

    assert {row.dataset_id for row in rows} == {"toy_reg"}
    assert not (tmp_path / "eval_out" / "failures.yaml").exists()
