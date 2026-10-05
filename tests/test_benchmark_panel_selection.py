"""``BenchmarkPanelConfig.dataset_ids``: scoring a chosen subset of ``eval_root``."""

from __future__ import annotations

from pathlib import Path

import pytest

import remedi.evaluation.framework.tasks.benchmark as benchmark_module
from remedi.data_handling.bundle import (
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    LabelColumn,
)
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.learners import LinearLearnerConfig
from remedi.evaluation.framework.tasks.benchmark import BenchmarkPanelConfig


def _spec(dataset_id: str) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        source_kind="test",
        smiles=True,
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        evaluation=EvaluationSpec(
            metrics=[EvalMetric.mae],
            split_columns=["split"],
            default_split="split",
            split_group="stereoisomer_id",
        ),
    )


@pytest.fixture
def three_benchmarks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    found = [(tmp_path / name, _spec(name)) for name in ("esol", "HIA_Hou", "hiv")]
    monkeypatch.setattr(benchmark_module, "discover_benchmarks", lambda root: found)
    return tmp_path


def _panel(eval_root: Path, dataset_ids: list[str] | None) -> BenchmarkPanelConfig:
    return BenchmarkPanelConfig(
        eval_root=eval_root,
        learners=[LinearLearnerConfig(ridge_alpha=1.0)],
        dataset_ids=dataset_ids,
    )


@pytest.mark.parametrize(
    ("dataset_ids", "expected"),
    [
        (None, ["esol", "HIA_Hou", "hiv"]),
        # A selection keeps discovery order, not the order it was given in.
        (["hiv", "esol"], ["esol", "hiv"]),
    ],
)
def test_selected_benchmarks(
    three_benchmarks: Path, dataset_ids: list[str] | None, expected: list[str]
) -> None:
    selected = _panel(three_benchmarks, dataset_ids).selected_benchmarks()
    assert [spec.dataset_id for _, spec in selected] == expected


def test_an_unknown_dataset_id_is_an_error(three_benchmarks: Path) -> None:
    with pytest.raises(ValueError, match="no benchmark under"):
        _panel(three_benchmarks, ["esol", "tox21"]).selected_benchmarks()
