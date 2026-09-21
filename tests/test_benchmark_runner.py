"""End-to-end runner test over real, ingested benchmark zarrs.

Every zarr here is built by the actual ``ingest_benchmark`` prepare task from a
real bundle, so discovery, the copied ``benchmark.yaml``, the split read out of
``table.parquet`` and the runner's cross-product are all exercised on the same
artifacts a prepare run produces. Pins: one row per scored target, the metric
and the cell identity (``split_column`` / ``seed``) reach ``results.csv``, a
non-default split column can be selected without re-ingesting, and one failing
benchmark does not sink the panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from remedi.data_handling.bundle import BenchmarkTask
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.descriptors import EcfpConfig
from remedi.evaluation.benchmark.learners import LinearLearnerConfig
from remedi.evaluation.benchmark.runner import (
    BenchmarkResultRow,
    EvalConfig,
    run_eval,
)

from .helpers.bundle_fixtures import (
    ACHIRAL_TEN_SMILES,
    TEN_ROW_SPLIT,
    ingest_tiny_bundle,
    write_conformers_bundle,
)

N_ROWS = len(ACHIRAL_TEN_SMILES)
#: A second, deliberately differently *sized* partition that exists only in the
#: table, never in the zarr's ``tasks/split`` array: two train, two valid, six
#: test, so the fold counts alone say which column was read.
ALTERNATIVE_SPLIT = ["train"] * 2 + ["valid"] * 2 + ["test"] * 6


def build_zarr(
    tmp_path: Path,
    dataset_id: str,
    *,
    tasks: list[BenchmarkTask],
    metrics: list[str],
    targets: np.ndarray,
    splits: dict[str, list[str]] | None = None,
) -> Path:
    """One bundle -> one ingested zarr under ``tmp_path/datasets``."""
    write_conformers_bundle(
        tmp_path / "bundles",
        dataset_id=dataset_id,
        tasks=tasks,
        metrics=metrics,
        targets=targets,
        splits=splits,
    )
    return ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "datasets", dataset_id)


def regression_zarr(tmp_path: Path, dataset_id: str = "toy_reg", **kwargs) -> Path:
    return build_zarr(
        tmp_path,
        dataset_id,
        tasks=[BenchmarkTask(name="y", task_type=TaskType.regression)],
        metrics=["RMSE"],
        targets=np.linspace(0.0, 1.0, N_ROWS),
        **kwargs,
    )


def eval_config(tmp_path: Path, **overrides) -> EvalConfig:
    defaults = dict(
        eval_root=tmp_path / "datasets",
        output_dir=tmp_path / "eval_out",
        descriptors=[EcfpConfig(name="ecfp_512", length=512)],
        learners=[LinearLearnerConfig(ridge_alpha=1.0)],
    )
    return EvalConfig(**{**defaults, **overrides})


# ------------------------------------------------------------------ the cells


def test_run_eval_regression_writes_a_single_row(tmp_path: Path) -> None:
    regression_zarr(tmp_path)

    rows = run_eval(eval_config(tmp_path))

    (row,) = rows
    assert isinstance(row, BenchmarkResultRow)
    assert row.dataset_id == "toy_reg"
    assert row.descriptor_name == "ecfp_512"
    assert row.learner_kind == "linear"
    assert row.metric_name == "RMSE"
    assert np.isfinite(row.metric_value)
    assert (row.n_train, row.n_val, row.n_test) == (6, 2, 2)
    # The spec's own default_split, and the config's seed.
    assert (row.split_column, row.seed) == ("split", 0)

    csv = pd.read_csv(tmp_path / "eval_out" / "results.csv")
    assert len(csv) == 1
    assert "source" not in csv.columns
    assert set(csv["split_column"]) == {"split"}
    assert (tmp_path / "eval_out" / "results.yaml").exists()


def test_run_eval_binary_classification_dispatch(tmp_path: Path) -> None:
    # Alternating labels so the two test rows carry both classes — otherwise
    # sklearn's AUROC raises (only-one-class-in-y_true).
    build_zarr(
        tmp_path,
        "toy_cls",
        tasks=[BenchmarkTask(name="y", task_type=TaskType.classification)],
        metrics=["AUROC"],
        targets=np.array([0.0, 1.0] * (N_ROWS // 2)),
    )

    rows = run_eval(
        eval_config(
            tmp_path,
            descriptors=[EcfpConfig(name="ecfp_256", length=256)],
            learners=[LinearLearnerConfig(logreg_C=1.0)],
        )
    )

    (row,) = rows
    assert row.metric_name == "AUROC"
    assert np.isfinite(row.metric_value)


def test_run_eval_cross_product_of_descriptors_and_learners(tmp_path: Path) -> None:
    regression_zarr(tmp_path)

    rows = run_eval(
        eval_config(
            tmp_path,
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
    assert {(row.descriptor_name, row.learner_kind) for row in rows} == {
        ("ecfp_128", "linear"),
        ("ecfp_512", "linear"),
    }


def test_run_eval_multitarget_regression_one_row_per_column(tmp_path: Path) -> None:
    """Three target columns give three rows, and NaN labels are dropped, not fatal."""
    generator = np.random.default_rng(0)
    targets = generator.normal(size=(N_ROWS, 3))
    # Punch holes in the train fold of column 1 to exercise the NaN filtering.
    targets[0, 1] = np.nan
    targets[3, 1] = np.nan
    build_zarr(
        tmp_path,
        "toy_multireg",
        tasks=[
            BenchmarkTask(name=f"y{index}", task_type=TaskType.regression)
            for index in range(3)
        ],
        metrics=["MAE"],
        targets=targets,
    )

    rows = run_eval(
        eval_config(tmp_path, descriptors=[EcfpConfig(name="ecfp_256", length=256)])
    )

    assert len(rows) == 3
    assert {row.target_col for row in rows} == {"y0", "y1", "y2"}
    assert all(np.isfinite(row.metric_value) for row in rows)
    assert all(row.metric_name == "MAE" for row in rows)
    csv = pd.read_csv(tmp_path / "eval_out" / "results.csv")
    assert set(csv["target_col"]) == {"y0", "y1", "y2"}
    assert not (tmp_path / "eval_out" / "failures.yaml").exists()


# ------------------------------------------------------------- split columns


def test_a_non_default_split_column_is_read_from_the_table(tmp_path: Path) -> None:
    """The zarr stores only the default split; the rest come from the table."""
    regression_zarr(
        tmp_path,
        splits={"split": list(TEN_ROW_SPLIT), "split__seed1": ALTERNATIVE_SPLIT},
    )

    default_row, *_ = run_eval(eval_config(tmp_path))
    alternative_row, *_ = run_eval(
        eval_config(
            tmp_path,
            output_dir=tmp_path / "eval_out_seed1",
            split_column="split__seed1",
        )
    )

    assert (default_row.n_train, default_row.n_val, default_row.n_test) == (6, 2, 2)
    assert (
        alternative_row.n_train,
        alternative_row.n_val,
        alternative_row.n_test,
    ) == (2, 2, 6)
    assert alternative_row.split_column == "split__seed1"


def test_an_unknown_split_column_fails_that_benchmark_only(tmp_path: Path) -> None:
    regression_zarr(tmp_path)

    rows = run_eval(eval_config(tmp_path, split_column="split__nope"))

    assert rows == []
    failures = (tmp_path / "eval_out" / "failures.yaml").read_text()
    assert "split__nope" in failures


def test_the_seed_reaches_every_row(tmp_path: Path) -> None:
    regression_zarr(tmp_path)

    rows = run_eval(eval_config(tmp_path, seed=7))

    assert all(row.seed == 7 for row in rows)


# ---------------------------------------------------------- fault tolerance


def test_run_eval_keeps_going_when_one_benchmark_fails(tmp_path: Path) -> None:
    """A mixed-task-type benchmark is recorded in failures.yaml and skipped."""
    regression_zarr(tmp_path, dataset_id="toy_ok")
    build_zarr(
        tmp_path,
        "toy_mixed",
        tasks=[
            BenchmarkTask(name="r", task_type=TaskType.regression),
            BenchmarkTask(name="c", task_type=TaskType.classification),
        ],
        metrics=["RMSE"],
        targets=np.stack(
            [np.linspace(0.0, 1.0, N_ROWS), (np.arange(N_ROWS) % 2).astype(float)],
            axis=1,
        ),
    )

    rows = run_eval(eval_config(tmp_path, descriptors=[EcfpConfig(name="ecfp_128")]))

    assert {row.dataset_id for row in rows} == {"toy_ok"}
    csv = pd.read_csv(tmp_path / "eval_out" / "results.csv")
    assert set(csv["dataset_id"]) == {"toy_ok"}
    assert "toy_mixed" in (tmp_path / "eval_out" / "failures.yaml").read_text()


def test_discovery_ignores_the_prepare_runs_own_directories(tmp_path: Path) -> None:
    """``output_root`` also holds ``status.yaml`` and the per-task summaries."""
    regression_zarr(tmp_path)
    (tmp_path / "datasets" / "ingest_benchmark").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "status.yaml").write_text("n_tasks: 1\n")

    rows = run_eval(eval_config(tmp_path))

    assert {row.dataset_id for row in rows} == {"toy_reg"}
