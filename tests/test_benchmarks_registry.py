"""The pydantic benchmark registry: counts, TaskSet mapping, yaml round-trip."""

from __future__ import annotations

import pydantic
import pydantic_yaml as pyd_yaml
import pytest

from threedscriptors.data_handling.benchmarks import (
    MOLECULENET_BENCHMARKS,
    TDC_BENCHMARKS,
    Benchmark,
    MoleculeNetBenchmark,
    TdcBenchmark,
    get_benchmark,
    select_benchmarks,
)
from threedscriptors.data_handling.dataset.tasks import TaskType


def test_panel_counts() -> None:
    assert len(MOLECULENET_BENCHMARKS) == 10
    assert len(TDC_BENCHMARKS) == 22
    ids = {b.dataset_id for b in MOLECULENET_BENCHMARKS}
    # Junior's 8 + the two the user asked to add back.
    assert {"hiv", "bace_pic50"}.issubset(ids)


def test_multilabel_modelled_as_classification_columns() -> None:
    sider = get_benchmark("sider")
    assert sider.metric.value == "macro-AUROC"
    assert len(sider.tasks) == 27
    assert all(t.task_type is TaskType.classification for t in sider.tasks)
    tox21 = get_benchmark("tox21")
    assert len(tox21.tasks) == 12


def test_task_set_roundtrip() -> None:
    clintox = get_benchmark("clintox")
    ts = clintox.task_set()
    assert [c.name for c in ts.system_cols] == ["FDA_APPROVED", "CT_TOX"]
    assert ts.system_map == {"FDA_APPROVED": 0, "CT_TOX": 1}


def test_source_column_defaults_to_name() -> None:
    esol = get_benchmark("esol")
    (task,) = esol.tasks
    assert task.name == "solubility"
    assert task.column == "measured log solubility in mols per litre"
    sider_task = get_benchmark("sider").tasks[0]
    assert sider_task.column == sider_task.name  # no rename for multilabel cols


def test_discriminated_union_parses_by_source() -> None:
    adapter = pydantic.TypeAdapter(Benchmark)
    mn = adapter.validate_python(MOLECULENET_BENCHMARKS[0].model_dump())
    assert isinstance(mn, MoleculeNetBenchmark)
    tdc = adapter.validate_python(TDC_BENCHMARKS[0].model_dump())
    assert isinstance(tdc, TdcBenchmark)


def test_yaml_config_roundtrip(tmp_path) -> None:
    src = get_benchmark("lipophilicity")
    path = tmp_path / "bench.yaml"
    pyd_yaml.to_yaml_file(path, src)
    loaded = pyd_yaml.parse_yaml_file_as(MoleculeNetBenchmark, path)
    assert loaded == src


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_benchmark("not_a_dataset")


def test_select_benchmarks_filters_and_limits() -> None:
    picked = select_benchmarks(MOLECULENET_BENCHMARKS, ids=["esol", "tox21"])
    assert {b.dataset_id for b in picked} == {"esol", "tox21"}
    assert len(select_benchmarks(TDC_BENCHMARKS, limit=3)) == 3
