"""Discovering ingested benchmark zarrs by the spec copied into them.

``ingest_benchmark`` copies ``benchmark.yaml`` into the zarr directory, which
is what makes a prepared zarr self-describing: the eval side reads the metrics,
the tasks and the split columns from there and never imports a registry
(``BENCHMARK_DATA_FORMAT.md`` §1.3). This module pins that contract —
:func:`discover_benchmark_zarrs` must find exactly the directories that carry
*both* a spec and a ``dataset_config.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from remedi.data_handling.bundle import (
    DATASET_CONFIG_FILENAME,
    SPEC_FILENAME,
    BenchmarkSpec,
    BenchmarkTask,
    BundleValidationError,
    EvalMetric,
    discover_benchmark_zarrs,
    read_spec,
)
from remedi.data_handling.dataset.tasks import TaskType


def make_spec(dataset_id: str = "esol") -> BenchmarkSpec:
    return BenchmarkSpec(
        dataset_id=dataset_id,
        description="A synthetic spec for the discovery tests.",
        tasks=[BenchmarkTask(name="measured", task_type=TaskType.regression)],
        metrics=[EvalMetric.rmse, EvalMetric.mae],
        stage="conformers",
        geometry_origin="etkdg_mmff",
        split_columns=["split", "split__seed1"],
        default_split="split",
        split_group="stereoisomer_id",
        source_kind="synthetic",
    )


def write_zarr_dir(
    root: Path,
    dataset_id: str,
    *,
    with_spec: bool = True,
    with_dataset_config: bool = True,
) -> Path:
    """A directory that looks like an ingested zarr to discovery, and nothing more."""
    directory = root / dataset_id
    directory.mkdir(parents=True)
    if with_spec:
        (directory / SPEC_FILENAME).write_text(
            yaml.safe_dump(make_spec(dataset_id).model_dump(mode="json"))
        )
    if with_dataset_config:
        (directory / DATASET_CONFIG_FILENAME).write_text("contains_smiles: false\n")
    return directory


# ------------------------------------------------------------- the round trip


def test_spec_round_trips_through_a_zarr_directory(tmp_path: Path) -> None:
    directory = write_zarr_dir(tmp_path, "esol")

    spec = read_spec(directory)

    assert spec == make_spec("esol")
    assert spec.metrics[0] is EvalMetric.rmse
    assert spec.task_names() == ["measured"]
    assert spec.default_split in spec.split_columns


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


def test_discovery_finds_every_ingested_zarr_sorted(tmp_path: Path) -> None:
    write_zarr_dir(tmp_path, "zeta")
    write_zarr_dir(tmp_path, "alpha")

    discovered = discover_benchmark_zarrs(tmp_path)

    assert [path.name for path, _ in discovered] == ["alpha", "zeta"]
    assert [spec.dataset_id for _, spec in discovered] == ["alpha", "zeta"]


def test_discovery_skips_a_directory_without_a_spec(tmp_path: Path) -> None:
    write_zarr_dir(tmp_path, "good")
    write_zarr_dir(tmp_path, "pretraining_zarr", with_spec=False)

    assert [path.name for path, _ in discover_benchmark_zarrs(tmp_path)] == ["good"]


def test_discovery_skips_a_bundle_directory(tmp_path: Path) -> None:
    # A bundle has a benchmark.yaml but no dataset_config.yaml; pointing
    # eval_root at a benchmark_root must therefore find nothing rather than
    # crash on a directory that holds no zarr arrays.
    write_zarr_dir(tmp_path, "bundle_only", with_dataset_config=False)

    assert discover_benchmark_zarrs(tmp_path) == []


def test_discovery_of_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert discover_benchmark_zarrs(tmp_path / "nothing_here") == []
