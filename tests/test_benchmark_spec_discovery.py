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
    BundleValidationError,
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


# ------------------------------------------------------------- the round trip


def test_spec_round_trips_through_a_dataset_directory(tmp_path: Path) -> None:
    directory = write_zarr_dir(tmp_path, "esol")

    spec = read_spec(directory)

    assert spec == make_spec("esol")
    evaluation = evaluation_of(spec)
    assert evaluation.metrics[0] is EvalMetric.rmse
    assert spec.label_names() == ["measured"]
    assert evaluation.default_split in evaluation.split_columns


def test_evaluation_of_a_corpus_names_the_problem() -> None:
    with pytest.raises(ValueError, match="no evaluation block"):
        evaluation_of(make_spec("corpus", evaluation=False))


def test_read_spec_refuses_an_unsupported_format_version(tmp_path: Path) -> None:
    directory = tmp_path / "future"
    directory.mkdir()
    document = make_spec().model_dump(mode="json")
    document["format_version"] = 2
    (directory / SPEC_FILENAME).write_text(yaml.safe_dump(document))

    with pytest.raises(BundleValidationError):
        read_spec(directory)


def test_read_spec_without_a_spec_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_spec(tmp_path)


# --------------------------------------------------------------- the discovery


def test_discovery_finds_every_benchmark_sorted(tmp_path: Path) -> None:
    write_zarr_dir(tmp_path, "zeta")
    write_zarr_dir(tmp_path, "alpha")

    discovered = discover_benchmarks(tmp_path)

    assert [path.name for path, _ in discovered] == ["alpha", "zeta"]
    assert [spec.dataset_id for _, spec in discovered] == ["alpha", "zeta"]


def test_discovery_skips_a_corpus_but_lists_it_as_a_dataset(tmp_path: Path) -> None:
    write_zarr_dir(tmp_path, "benchmark")
    write_zarr_dir(tmp_path, "corpus", evaluation=False)

    assert [path.name for path, _ in discover_benchmarks(tmp_path)] == ["benchmark"]
    assert [path.name for path, _ in discover_datasets(tmp_path)] == [
        "benchmark",
        "corpus",
    ]


def test_discovery_skips_a_directory_without_a_spec(tmp_path: Path) -> None:
    write_zarr_dir(tmp_path, "good")
    write_zarr_dir(tmp_path, "pretraining_zarr", with_spec=False)

    assert [path.name for path, _ in discover_benchmarks(tmp_path)] == ["good"]


def test_discovery_skips_a_bundle_directory(tmp_path: Path) -> None:
    # A bundle has a dataset.yaml but no dataset_config.yaml; pointing
    # eval_root at a bundle_root must therefore find nothing rather than crash
    # on a directory that holds no zarr arrays.
    write_zarr_dir(tmp_path, "bundle_only", with_dataset_config=False)

    assert discover_benchmarks(tmp_path) == []


def test_discovery_of_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert discover_benchmarks(tmp_path / "nothing_here") == []
