"""bundle -> zarr: the ids survive the ingest, and ``verify_benchmark`` says so.

``BENCHMARK_DATA_FORMAT.md`` §2.3 is the point of this module. Today's failure
class is that the id a split was computed against and the id written into the
zarr come from different code paths; after the ingest task they must be the
same numbers, and ``ids/bundle_row`` must make every zarr row joinable back to
``table.parquet`` (which §4.1's predictions cache is gated on).

Everything here goes through the real ``ingest_benchmark`` and
``verify_benchmark`` tasks — nothing hand-writes a zarr array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from remedi.data_handling.bundle import (
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    TABLE_FILENAME,
    BenchmarkTask,
    content_hash_of_table,
    read_bundle,
    read_table,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import TaskType, split_codes_from_names
from remedi.data_handling.prepare import (
    IngestBenchmarkConfig,
    PrepareContext,
    VerifyBenchmarkConfig,
)

from .helpers.bundle_fixtures import (
    ACHIRAL_TEN_SMILES,
    TEN_ROW_SPLIT,
    ingest_tiny_bundle,
    write_conformers_bundle,
    write_smiles_bundle,
)

DATASET_ID = "toy_reg"


def build_bundle(tmp_path: Path, **kwargs) -> Path:
    """A ten-row regression bundle under ``tmp_path/bundles``."""
    defaults = dict(
        dataset_id=DATASET_ID,
        tasks=[BenchmarkTask(name="y", task_type=TaskType.regression)],
        metrics=["RMSE"],
        targets=np.linspace(0.0, 1.0, len(ACHIRAL_TEN_SMILES)),
    )
    return write_conformers_bundle(tmp_path / "bundles", **{**defaults, **kwargs})


def make_context(tmp_path: Path) -> PrepareContext:
    return PrepareContext(
        smiles_bundle_root=tmp_path / "bundles",
        benchmark_root=tmp_path / "bundles",
        output_root=tmp_path / "zarrs",
    )


def open_zarr(path: Path) -> MoleculeDataset:
    return MoleculeDataset.open_existing_dataset_from_dir(path)


# ------------------------------------------------------------- the id arrays


def test_every_id_array_equals_its_bundle_column(tmp_path: Path) -> None:
    bundle_directory = build_bundle(tmp_path)
    zarr_path = ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "zarrs", DATASET_ID)
    table = read_bundle(bundle_directory).table

    dataset = open_zarr(zarr_path)

    assert dataset.N_structures == len(table)
    np.testing.assert_array_equal(
        np.asarray(dataset.bundle_row[:]), table["structure_id"].to_numpy()
    )
    np.testing.assert_array_equal(
        np.asarray(dataset.molecule_ids[:]), table["molecule_id"].to_numpy()
    )
    np.testing.assert_array_equal(
        np.asarray(dataset.isomer_ids[:]), table["stereoisomer_id"].to_numpy()
    )
    # structure_id stays the row index, which the training-side splitter reads
    # positionally.
    np.testing.assert_array_equal(
        np.asarray(dataset.structure_ids[:]), np.arange(len(table))
    )


def test_the_split_array_is_the_default_split_column_in_codes(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    zarr_path = ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "zarrs", DATASET_ID)

    dataset = open_zarr(zarr_path)

    np.testing.assert_array_equal(
        np.asarray(dataset.split[:]), split_codes_from_names(TEN_ROW_SPLIT)
    )


def test_smiles_per_structure_comes_from_the_copied_table(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    zarr_path = ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "zarrs", DATASET_ID)

    dataset = open_zarr(zarr_path)

    # No SMILES store at all — contains_smiles is False for a benchmark zarr.
    assert dataset.config.contains_smiles is False
    assert dataset.smiles is None
    expected = read_table(zarr_path, columns=["isomeric_smiles"])
    assert dataset.get_smiles_per_structure() == expected["isomeric_smiles"].tolist()


def test_targets_and_masks_carry_the_tables_nulls(tmp_path: Path) -> None:
    targets = np.linspace(0.0, 1.0, len(ACHIRAL_TEN_SMILES))
    targets[3] = np.nan
    build_bundle(tmp_path, targets=targets)
    zarr_path = ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "zarrs", DATASET_ID)

    dataset = open_zarr(zarr_path)

    mask = np.asarray(dataset.mask_system[:]).ravel()
    np.testing.assert_array_equal(mask, (~np.isnan(targets)).astype(np.uint8))


# --------------------------------------------------------- the copied bundle


def test_the_three_bundle_files_travel_with_the_zarr(tmp_path: Path) -> None:
    bundle_directory = build_bundle(tmp_path)
    zarr_path = ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "zarrs", DATASET_ID)

    for filename in (SPEC_FILENAME, TABLE_FILENAME, PROVENANCE_FILENAME):
        assert (zarr_path / filename).is_file()
    assert yaml.safe_load((zarr_path / SPEC_FILENAME).read_text()) == yaml.safe_load(
        (bundle_directory / SPEC_FILENAME).read_text()
    )


def test_the_ingest_summary_records_the_content_hash_and_the_splits(
    tmp_path: Path,
) -> None:
    bundle_directory = build_bundle(tmp_path)
    task = IngestBenchmarkConfig(dataset_ids=[DATASET_ID], molecule_chunk=4)

    (result,) = list(task.run(make_context(tmp_path)))

    summary = result.obj  # type: ignore[attr-defined]
    assert summary.dataset_id == DATASET_ID
    assert summary.rows == len(ACHIRAL_TEN_SMILES)
    assert summary.n_atoms > 0
    assert summary.task_names == ["y"]
    assert summary.default_split == "split"
    assert summary.rows_per_split == {"train": 6, "valid": 2, "test": 2}
    assert summary.content_sha256 == content_hash_of_table(
        read_bundle(bundle_directory).table
    )
    assert summary.skipped is False


def test_a_second_ingest_skips_unless_overwrite_is_set(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    context = make_context(tmp_path)
    list(IngestBenchmarkConfig(dataset_ids=[DATASET_ID]).run(context))

    (skipped,) = list(IngestBenchmarkConfig(dataset_ids=[DATASET_ID]).run(context))
    assert skipped.obj.skipped is True  # type: ignore[attr-defined]

    (rebuilt,) = list(
        IngestBenchmarkConfig(dataset_ids=[DATASET_ID], overwrite=True).run(context)
    )
    assert rebuilt.obj.skipped is False  # type: ignore[attr-defined]
    assert rebuilt.obj.rows == len(ACHIRAL_TEN_SMILES)  # type: ignore[attr-defined]


def test_ingesting_a_smiles_stage_bundle_is_refused(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id="smiles_only")
    task = IngestBenchmarkConfig(dataset_ids=["smiles_only"])

    with pytest.raises(ValueError, match="only a 'conformers'-stage bundle"):
        list(task.run(make_context(tmp_path)))


# ------------------------------------------------------------ verification


def test_verify_benchmark_passes_on_a_freshly_ingested_zarr(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    context = make_context(tmp_path)
    list(IngestBenchmarkConfig(dataset_ids=[DATASET_ID]).run(context))

    (result,) = list(VerifyBenchmarkConfig(dataset_ids=[DATASET_ID]).run(context))

    report = result.obj  # type: ignore[attr-defined]
    assert report.ok is True
    assert report.problems == []
    assert report.rows == report.frames == report.structures_in_zarr
    assert report.zarr_path == tmp_path / "zarrs" / DATASET_ID


def test_verify_benchmark_checks_the_bundle_alone_before_any_ingest(
    tmp_path: Path,
) -> None:
    build_bundle(tmp_path)
    task = VerifyBenchmarkConfig(dataset_ids=[DATASET_ID])

    (result,) = list(task.run(make_context(tmp_path)))

    report = result.obj  # type: ignore[attr-defined]
    assert report.ok is True
    assert report.zarr_path is None
    assert report.structures_in_zarr == 0


def test_verify_benchmark_catches_a_stale_copied_table(tmp_path: Path) -> None:
    """The idempotent skip's failure mode: the zarr kept an older bundle."""
    build_bundle(tmp_path)
    context = make_context(tmp_path)
    list(IngestBenchmarkConfig(dataset_ids=[DATASET_ID]).run(context))
    # Rewrite the bundle's labels without re-ingesting.
    build_bundle(tmp_path, targets=np.linspace(5.0, 6.0, len(ACHIRAL_TEN_SMILES)))

    task = VerifyBenchmarkConfig(dataset_ids=[DATASET_ID])
    artifacts = []
    # The report is serialised as it is yielded, so it is on disk even though
    # the task then fails — which is the whole point of reporting before raising.
    with pytest.raises(ValueError, match="verify_benchmark found problems"):
        for result in task.run(context):
            artifacts.append(result.serialize_to(context.output_root))

    (artifact,) = artifacts
    report = yaml.safe_load((context.output_root / artifact["file_name"]).read_text())
    assert report["ok"] is False
    assert any("stale zarr" in problem for problem in report["problems"])
