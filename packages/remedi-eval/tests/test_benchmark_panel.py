"""End-to-end ``benchmark_panel`` runs over datasets written by the real ``write_dataset``.

Pins: one row per scored target, the cell identity (``split_column`` / ``seed``)
reaches ``results.csv``, a non-default split column is read from the table, a
corpus is not scored, and one failing benchmark does not sink the panel.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import pytest
import yaml
from remedi_data.bundle import EvalMetric, LabelColumn, TaskType

from remedi_eval.benchmark.learners import Learner, LinearLearner
from remedi_eval.descriptors import DescriptorConfig, EcfpConfig
from remedi_eval.framework import (
    BenchmarkPanelConfig,
    EvalManifest,
    run_manifest,
)
from testing import (
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


def run_panel(
    tmp_path: Path,
    *,
    descriptors: list[DescriptorConfig] | None = None,
    learners: list[Learner] | None = None,
    split_column: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run one panel over ``tmp_path/datasets``; its ``results.csv`` as a frame.

    The first descriptor is the run's model, the rest the panel's baselines.
    """
    model, *baselines = descriptors or [
        EcfpConfig(name="ecfp_128", fingerprint_size=128)
    ]
    run_manifest(
        EvalManifest(
            model=model,
            output_root=tmp_path / "eval_out",
            seed=seed,
            tasks=[
                BenchmarkPanelConfig(
                    eval_root=tmp_path / "datasets",
                    learners=learners or [LinearLearner(ridge_alpha=1.0)],
                    baseline_descriptors=baselines,
                    split_column=split_column,
                )
            ],
        )
    )
    try:
        return pd.read_csv(tmp_path / "eval_out" / "benchmark" / "results.csv")
    except pd.errors.EmptyDataError:  # every benchmark failed: a header-less file
        return pd.DataFrame()


def failures_text(tmp_path: Path) -> str:
    failures = tmp_path / "eval_out" / "benchmark" / "failures.yaml"
    return failures.read_text() if failures.exists() else ""


def _with_nan_holes(targets: np.ndarray) -> np.ndarray:
    """Holes in the train fold of column 1: NaN labels are dropped, not fatal."""
    targets = targets.copy()
    targets[[0, 3], 1] = np.nan
    return targets


# ------------------------------------------------------------------ the cells


def test_panel_cross_product_and_cell_identity(tmp_path: Path) -> None:
    build_zarr(tmp_path)

    rows = run_panel(
        tmp_path,
        seed=7,
        descriptors=[
            EcfpConfig(name="ecfp_128", fingerprint_size=128),
            EcfpConfig(name="ecfp_512", fingerprint_size=512),
        ],
        learners=[
            LinearLearner(name="ridge_weak", ridge_alpha=0.1),
            LinearLearner(name="ridge_strong", ridge_alpha=10.0),
        ],
    )

    # 1 benchmark x 2 descriptors x 2 learners = 4 rows.
    assert len(rows) == 4
    assert set(rows["descriptor_name"]) == {"ecfp_128", "ecfp_512"}
    identity = ["dataset_id", "learner_kind", "split_column", "seed"]
    counts = ["n_train", "n_val", "n_test"]
    # The spec's own default_split, and the manifest's seed.
    assert rows[identity].drop_duplicates().values.tolist() == [
        ["toy_reg", "linear", "split", 7]
    ]
    assert rows[counts].drop_duplicates().values.tolist() == [[6, 2, 2]]


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
def test_panel_scores_one_row_per_target(
    tmp_path: Path,
    labels: list[LabelColumn],
    metric: EvalMetric,
    targets: np.ndarray | None,
    expected_target_columns: set[str | None],
) -> None:
    build_zarr(tmp_path, labels=labels, metrics=(metric,), targets=targets)

    rows = run_panel(tmp_path)

    assert len(rows) == len(expected_target_columns)
    target_columns = {None if pd.isna(name) else name for name in rows["target_col"]}
    assert target_columns == expected_target_columns
    assert set(rows["metric_name"]) == {metric.value}
    assert np.isfinite(rows["metric_value"]).all()
    assert failures_text(tmp_path) == ""


# ------------------------------------------------------------- split columns


def test_split_column_selection(tmp_path: Path) -> None:
    """The zarr stores only the default split; the rest come from the table, and
    an unknown column fails that benchmark into ``failures.yaml``."""
    build_zarr(
        tmp_path,
        splits={"split": list(TEN_ROW_SPLIT), "split__seed1": ALTERNATIVE_SPLIT},
    )

    (row,) = run_panel(tmp_path, split_column="split__seed1").to_dict("records")
    assert (row["n_train"], row["n_val"], row["n_test"]) == (2, 2, 6)
    assert row["split_column"] == "split__seed1"

    assert run_panel(tmp_path, split_column="split__nope").empty
    assert "split__nope" in failures_text(tmp_path)


# ---------------------------------------------------------- fault tolerance


def test_panel_keeps_going_when_one_benchmark_fails(tmp_path: Path) -> None:
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

    rows = run_panel(tmp_path)

    assert set(rows["dataset_id"]) == {"toy_ok"}
    assert "toy_mixed" in failures_text(tmp_path)


def test_discovery_scores_only_benchmarks(tmp_path: Path) -> None:
    """A corpus (no evaluation block) and the build run's own directories and
    ``status.yaml`` under ``zarr_root`` are skipped, not failures."""
    build_zarr(tmp_path)
    build_zarr(tmp_path, "toy_corpus", metrics=None)
    (tmp_path / "datasets" / "build").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "verify").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "status.yaml").write_text("n_tasks: 1\n")

    rows = run_panel(tmp_path)

    assert set(rows["dataset_id"]) == {"toy_reg"}
    assert failures_text(tmp_path) == ""


# ------------------------------------------------------------ learner names


def test_learners_of_one_kind_are_told_apart_by_name(tmp_path: Path) -> None:
    build_zarr(tmp_path)

    rows = run_panel(
        tmp_path,
        learners=[
            LinearLearner(name="ridge_weak", ridge_alpha=0.1),
            LinearLearner(name="ridge_strong", ridge_alpha=10.0),
        ],
    )

    assert sorted(rows["learner_name"]) == ["ridge_strong", "ridge_weak"]
    assert set(rows["learner_kind"]) == {"linear"}


def test_learner_names_must_be_unique_across_the_panel(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError, match="linear"):
        BenchmarkPanelConfig(
            eval_root=tmp_path,
            learners=[LinearLearner(), LinearLearner(ridge_alpha=2.0)],
        )


# ----------------------------------------------------- failures, incrementally


def test_failures_are_written_before_the_panel_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job killed mid-panel still leaves the failures recorded so far."""
    import remedi_eval.framework.tasks.benchmark as benchmark_task

    build_zarr(tmp_path, "toy_a", splits={"split": list(TEN_ROW_SPLIT)})
    build_zarr(tmp_path, "toy_b")
    original = benchmark_task.evaluate_cell

    def fail_then_get_killed(*args, **kwargs):
        cell = inspect.signature(original).bind(*args, **kwargs).arguments["cell"]
        if cell.dataset_id == "toy_a":
            raise ValueError("toy_a cell broke")
        if cell.dataset_id == "toy_b":
            raise SystemExit("wall-clock limit")
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark_task, "evaluate_cell", fail_then_get_killed)
    with pytest.raises(SystemExit):
        run_panel(tmp_path)

    assert "toy_a cell broke" in failures_text(tmp_path)


def test_a_failing_cell_does_not_lose_the_other_cells_of_its_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import remedi_eval.framework.tasks.benchmark as benchmark_task

    build_zarr(tmp_path)
    original = benchmark_task.evaluate_cell

    def fail_for_one_learner(*args, **kwargs):
        arguments = inspect.signature(original).bind(*args, **kwargs).arguments
        learner = arguments["learner"]
        if learner.name == "broken":
            raise ValueError("this learner broke")
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark_task, "evaluate_cell", fail_for_one_learner)
    rows = run_panel(
        tmp_path,
        learners=[
            LinearLearner(name="broken"),
            LinearLearner(name="fine"),
        ],
    )

    assert list(rows["learner_name"]) == ["fine"]
    failures = yaml.safe_load(failures_text(tmp_path))["failures"]
    assert [
        (failure["dataset_id"], failure["descriptor_name"], failure["learner_name"])
        for failure in failures
    ] == [("toy_reg", "ecfp_128", "broken")]


# --------------------------------------------------------------- the protocol


def test_a_table_read_error_is_not_reported_as_a_missing_split_column(
    tmp_path: Path,
) -> None:
    from remedi_data.bundle import TABLE_FILENAME

    from remedi_data.dataset.molecule_dataset import MoleculeDataset
    from remedi_eval.protocol import read_partition

    dataset_dir = build_zarr(tmp_path)
    table_path = dataset_dir / TABLE_FILENAME
    pd.read_parquet(table_path).drop(columns=["structure_id"]).to_parquet(table_path)
    dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)

    with pytest.raises(ValueError, match="structure_id") as error:
        read_partition(dataset_dir, dataset, "split")
    assert "split column" not in str(error.value)

    with pytest.raises(ValueError, match="'split__nope' is not in"):
        read_partition(dataset_dir, dataset, "split__nope")


def test_without_keep_going_the_first_failure_is_recorded_and_raised(
    tmp_path: Path,
) -> None:
    build_zarr(tmp_path)
    manifest = EvalManifest(
        model=EcfpConfig(name="ecfp_128", fingerprint_size=128),
        output_root=tmp_path / "eval_out",
        keep_going=False,
        tasks=[
            BenchmarkPanelConfig(
                eval_root=tmp_path / "datasets",
                learners=[LinearLearner()],
                split_column="split__nope",
            )
        ],
    )

    with pytest.raises(ValueError, match="split__nope"):
        run_manifest(manifest)
    assert "split__nope" in failures_text(tmp_path)
