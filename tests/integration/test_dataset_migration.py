"""The one-off migration of datasets to the current identity and label layout.

``scripts/dataset_creation/migrate_dataset_identity.py`` re-hashes a dataset's
structures with ``total_charge`` and ``multiplicity`` and drops the zarr's
TaskSet and ``tasks/`` group. The old layout is simulated on datasets written
by the current writer: a TaskSet-style ``dataset_config.yaml``, a ``tasks/``
group holding the table's labels and default split, and the old three-array
structures hash in ``provenance.yaml``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

from remedi_data.build import structures_sha256, verify_dataset
from remedi_data.bundle import EvalMetric, LabelColumn, read_bundle, read_provenance
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.tasks import Split, TaskType, split_codes_from_names
from remedi_data.testing import write_small_dataset

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(relative_path: str):
    path = SCRIPTS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


migration = _load_script("dataset_creation/migrate_dataset_identity.py")

LABELS = [
    LabelColumn(name="active", task_type=TaskType.classification),
    LabelColumn(name="potency", task_type=TaskType.regression),
]
ZARR_ARRAYS = (
    "positions",
    "atomic_numbers",
    "molecule_ptr",
    "total_charge",
    "multiplicity",
    "ids/structure_id",
    "ids/molecule_id",
    "ids/stereoisomer_id",
)


def _benchmark(root: Path, dataset_id: str = "benchmark") -> Path:
    targets = np.column_stack(
        [np.tile([1.0, 0.0], 5), np.linspace(0.0, 9.0, 10)]
    ).astype(np.float64)
    targets[2, 0] = np.nan
    return write_small_dataset(
        root,
        dataset_id=dataset_id,
        labels=LABELS,
        metrics=[EvalMetric.macro_auroc],
        targets=targets,
    )


def _corpus(root: Path) -> Path:
    return write_small_dataset(
        root, dataset_id="corpus", labels=[], metrics=None, targets=None
    )


def _old_structures_sha256(directory: Path) -> str:
    """The definition before 2026-11-16: structures without charge and spin."""
    group = zarr.open_group(directory, mode="r")
    digest = hashlib.sha256()
    for name in ("atomic_numbers", "positions", "molecule_ptr"):
        values = np.ascontiguousarray(np.asarray(group[name][:]))
        digest.update(f"{name}:{values.dtype.str}:{values.shape}\n".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _write_old_layout(directory: Path, *, with_tasks: bool = True) -> None:
    """Turn a current dataset into one the old writer would have written."""
    bundle = read_bundle(directory)
    spec, table = bundle.spec, bundle.table
    config_path = directory / "dataset_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["contains_smiles"] = False
    config["tasks"] = None
    if with_tasks:
        config["tasks"] = {
            "atom_cols": [],
            "atom_map": {},
            "system_cols": [
                {
                    "auxillary_dim": None,
                    "name": label.name,
                    "scope": "system",
                    "task_type": label.task_type.value,
                }
                for label in spec.labels
            ],
            "system_map": {
                label.name: index for index, label in enumerate(spec.labels)
            },
        }
        values = table[spec.label_names()].to_numpy(dtype=np.float64)
        present = ~np.isnan(values)
        tasks = zarr.open_group(directory, mode="r+").create_group("tasks")
        tasks.create_array(
            "targets_system", data=np.where(present, values, 0.0).astype("f4")
        )
        tasks.create_array("mask_system", data=present.astype("u1"))
        split = (
            split_codes_from_names(table[spec.evaluation.default_split].tolist())
            if spec.evaluation is not None
            else np.full(len(table), Split.unassigned.value, dtype="u1")
        )
        tasks.create_array("split", data=split)
    config_path.write_text(yaml.safe_dump(config))
    provenance_path = directory / "provenance.yaml"
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["structures_sha256"] = _old_structures_sha256(directory)
    provenance_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    )


def _array_digests(directory: Path) -> dict[str, str]:
    group = zarr.open_group(directory, mode="r")
    return {
        name: hashlib.sha256(np.asarray(group[name][:]).tobytes()).hexdigest()
        for name in ZARR_ARRAYS
    }


def _file_digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_an_old_dataset_is_migrated_verified_and_left_alone_the_second_time(
    tmp_path: Path,
) -> None:
    directory = _benchmark(tmp_path)
    _write_old_layout(directory)
    table_before = (directory / "table.parquet").read_bytes()
    spec_before = (directory / "dataset.yaml").read_bytes()
    arrays_before = _array_digests(directory)
    with pytest.raises(ValueError, match="tasks"):  # the old config no longer opens
        MoleculeDataset.open_existing_dataset_from_dir(directory)

    first = migration.migrate([directory], dry_run=False, workers=1)

    (report,) = first.datasets
    assert report.outcome == "migrated", report.problems
    assert len(report.changes) == 3
    assert not (directory / "tasks").exists()
    config = yaml.safe_load((directory / "dataset_config.yaml").read_text())
    assert "tasks" not in config and "contains_smiles" not in config
    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    new_hash = structures_sha256(dataset)
    assert report.recorded_structures_sha256 == _old_structures_sha256(directory)
    assert report.structures_sha256 == new_hash
    outputs = read_provenance(directory).outputs
    assert outputs is not None and outputs.structures_sha256 == new_hash
    # Nothing else moved: arrays, table and spec are as they were.
    assert _array_digests(directory) == arrays_before
    assert (directory / "table.parquet").read_bytes() == table_before
    assert (directory / "dataset.yaml").read_bytes() == spec_before
    assert verify_dataset(directory).problems == []

    files_after_first = _file_digests(directory)
    second = migration.migrate([directory], dry_run=False, workers=1)
    assert [entry.outcome for entry in second.datasets] == ["unchanged"]
    assert _file_digests(directory) == files_after_first


def test_a_dataset_written_today_is_unchanged(tmp_path: Path) -> None:
    directory = _benchmark(tmp_path)
    files_before = _file_digests(directory)

    report = migration.migrate([tmp_path], dry_run=False, workers=1)

    assert [entry.outcome for entry in report.datasets] == ["unchanged"]
    assert report.datasets[0].changes == []
    assert _file_digests(directory) == files_before


def test_a_dry_run_reports_and_changes_nothing(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)
    corpus = _corpus(tmp_path)
    _write_old_layout(benchmark)
    _write_old_layout(corpus, with_tasks=False)
    before = {path: _file_digests(path) for path in (benchmark, corpus)}

    report = migration.migrate([tmp_path], dry_run=True, workers=1)

    assert [(entry.dataset_path, entry.outcome) for entry in report.datasets] == [
        (benchmark, "would_migrate"),
        (corpus, "would_migrate"),
    ]
    by_path = {entry.dataset_path: entry for entry in report.datasets}
    assert len(by_path[benchmark].changes) == 3
    # The corpus had no labels: its config loses the null TaskSet, its hash moves.
    assert len(by_path[corpus].changes) == 2
    assert {path: _file_digests(path) for path in (benchmark, corpus)} == before
    assert (benchmark / "tasks").is_dir()


def test_a_tasks_group_that_disagrees_with_the_table_is_kept_and_reported(
    tmp_path: Path,
) -> None:
    directory = _benchmark(tmp_path)
    _write_old_layout(directory)
    targets = zarr.open_array(directory / "tasks" / "targets_system", mode="r+")
    targets[0, 1] = 42.0
    files_before = _file_digests(directory)

    report = migration.migrate([directory], dry_run=False, workers=1)

    (entry,) = report.datasets
    assert entry.outcome == "failed"
    assert any("targets_system" in problem for problem in entry.problems)
    assert _file_digests(directory) == files_before


def test_a_path_without_datasets_is_refused(tmp_path: Path) -> None:
    with pytest.raises(migration.MigrationError, match="holds no datasets"):
        migration.migrate([tmp_path], dry_run=True, workers=1)


def test_the_command_line_writes_a_report_and_fails_on_a_failure(
    tmp_path: Path,
) -> None:
    directory = _benchmark(tmp_path / "datasets")
    _write_old_layout(directory)
    report_path = tmp_path / "report.yaml"

    assert (
        migration.main(
            [
                "--dry-run",
                "--workers",
                "1",
                "--report",
                str(report_path),
                str(tmp_path / "datasets"),
            ]
        )
        == 0
    )
    written = yaml.safe_load(report_path.read_text())
    assert written["dry_run"] is True
    assert [entry["outcome"] for entry in written["datasets"]] == ["would_migrate"]

    zarr.open_array(directory / "tasks" / "split", mode="r+")[:] = Split.train.value
    assert migration.main(["--workers", "1", str(directory)]) == 1
    assert (directory / "tasks").is_dir()


def _record_in_provenance(directory: Path, key: str, value: str) -> None:
    path = directory / "provenance.yaml"
    document = yaml.safe_load(path.read_text())
    document["outputs"]["table.parquet"][key] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False))


def test_a_moved_content_hash_is_re_recorded_when_the_table_bytes_are_unchanged(
    tmp_path: Path,
) -> None:
    directory = _benchmark(tmp_path)
    content = read_provenance(directory).outputs.table_parquet.content_sha256
    _record_in_provenance(directory, "content_sha256", "0" * 64)

    report = migration.migrate_dataset(directory, dry_run=False, workers=1)

    assert report.outcome == "migrated", report.problems
    assert any("content_sha256" in change for change in report.changes)
    assert read_provenance(directory).outputs.table_parquet.content_sha256 == content
    read_bundle(directory)


def test_a_changed_table_is_not_re_keyed(tmp_path: Path) -> None:
    directory = _benchmark(tmp_path)
    _record_in_provenance(directory, "content_sha256", "0" * 64)
    _record_in_provenance(directory, "file_sha256", "1" * 64)

    report = migration.migrate_dataset(directory, dry_run=False, workers=1)

    assert report.outcome == "failed"
    assert "file_sha256 differs" in " ".join(report.problems)
    assert read_provenance(directory).outputs.table_parquet.content_sha256 == "0" * 64
