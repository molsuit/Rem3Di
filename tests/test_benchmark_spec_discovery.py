"""Discovering benchmark datasets by the ``dataset.yaml`` beside their zarr.

A dataset is self-describing: the evaluation side reads the metrics, the labels
and the split columns from its ``dataset.yaml`` and never imports a registry.
This module pins that contract: :func:`discover_benchmarks` must find exactly
the directories that carry *both* a spec and a ``dataset_config.yaml`` *and*
whose spec declares an evaluation block.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from remedi.data_handling.bundle import (
    DATASET_CONFIG_FILENAME,
    SPEC_FILENAME,
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    LabelColumn,
    discover_datasets,
    read_spec,
)
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.runner import discover_benchmarks, evaluation_of


def make_spec(dataset_id: str = "esol", *, evaluation: bool = True) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        description="A synthetic spec for the discovery tests.",
        smiles=True,
        geometry_origin="etkdg_mmff",
        labels=[LabelColumn(name="measured", task_type=TaskType.regression)],
        evaluation=(
            EvaluationSpec(
                metrics=[EvalMetric.rmse, EvalMetric.mae],
                split_columns=["split", "split__seed1"],
                default_split="split",
                split_group="stereoisomer_id",
            )
            if evaluation
            else None
        ),
        source_kind="synthetic",
    )


def write_zarr_dir(
    root: Path,
    dataset_id: str,
    *,
    with_spec: bool = True,
    with_dataset_config: bool = True,
    evaluation: bool = True,
) -> Path:
    """A directory that looks like a dataset to discovery, and nothing more."""
    directory = root / dataset_id
    directory.mkdir(parents=True)
    if with_spec:
        spec = make_spec(dataset_id, evaluation=evaluation)
        (directory / SPEC_FILENAME).write_text(
            yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True))
        )
    if with_dataset_config:
        (directory / DATASET_CONFIG_FILENAME).write_text("contains_smiles: false\n")
    return directory


def test_the_evaluation_block_is_read_back_from_a_dataset_directory(
    tmp_path: Path,
) -> None:
    spec = read_spec(write_zarr_dir(tmp_path, "esol"))
    assert spec == make_spec("esol")
    evaluation = evaluation_of(spec)
    assert evaluation.metrics[0] is EvalMetric.rmse
    assert evaluation.default_split in evaluation.split_columns
    with pytest.raises(ValueError, match="no evaluation block"):
        evaluation_of(make_spec("corpus", evaluation=False))


def test_discovery_finds_only_benchmarks_sorted(tmp_path: Path) -> None:
    write_zarr_dir(tmp_path, "zeta")
    write_zarr_dir(tmp_path, "alpha")
    # a corpus is a dataset, but not a benchmark
    write_zarr_dir(tmp_path, "corpus", evaluation=False)
    write_zarr_dir(tmp_path, "pretraining_zarr", with_spec=False)
    # A bundle has a dataset.yaml but no dataset_config.yaml; pointing eval_root
    # at a bundle_root must therefore find nothing rather than crash on a
    # directory that holds no zarr arrays.
    write_zarr_dir(tmp_path, "bundle_only", with_dataset_config=False)

    discovered = discover_benchmarks(tmp_path)

    assert [path.name for path, _ in discovered] == ["alpha", "zeta"]
    assert [spec.dataset_id for _, spec in discovered] == ["alpha", "zeta"]
    assert [path.name for path, _ in discover_datasets(tmp_path)] == [
        "alpha",
        "corpus",
        "zeta",
    ]
    assert discover_benchmarks(tmp_path / "nothing_here") == []
