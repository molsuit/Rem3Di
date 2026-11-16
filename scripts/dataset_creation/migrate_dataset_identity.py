"""Bring datasets written before 2026-11-16 to the current layout, in place (one-off).

Three format changes, none of which touches coordinates or any other array the
model reads:

* ``content_sha256`` of ``table.parquet`` folds ``-0.0`` into ``0.0`` (pandas 3.0
  and 3.1 hash the two zeros differently). A recorded value that no longer
  matches is re-recorded, but only when the table's ``file_sha256`` proves its
  bytes are the ones recorded; otherwise the dataset fails.
* ``structures_sha256`` now also covers ``total_charge`` and ``multiplicity``.
  It is recomputed and written to ``provenance.yaml`` (``outputs``), the only
  place a dataset records it.
* Labels live only in ``dataset.yaml`` and ``table.parquet``. ``dataset_config.yaml``
  loses its ``tasks`` block (and any other key ``DatasetConfig`` does not know,
  such as an old ``contains_smiles``), and the zarr's ``tasks/`` group
  (``targets_system``, ``mask_system``, ``split``) is deleted, but only after
  checking that every value it holds equals the table's.

Then ``verify_dataset`` runs on the result. A dataset already in the current
layout is reported ``unchanged`` (and still verified), so the script is safe to
rerun. ``--dry-run`` migrates a shadow of each dataset instead (symbolic links
to its arrays and table, copies of the two yaml files it rewrites, in a
temporary directory) and verifies that, so nothing in the dataset changes.

Usage (a path is a dataset directory or a directory holding datasets)::

    .venv/bin/python scripts/dataset_creation/migrate_dataset_identity.py --dry-run \
        benchmark_data/datasets /dais/fs/scratch/wedigs/remedi/pretraining
    .venv/bin/python scripts/dataset_creation/migrate_dataset_identity.py \
        --report migration_report.yaml benchmark_data/datasets
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import zarr
from pydantic import BaseModel, ConfigDict, Field
from zarr.storage import LocalStore

from remedi_data.build import structures_sha256, verify_dataset
from remedi_data.bundle import (
    DATASET_CONFIG_FILENAME,
    PROVENANCE_FILENAME,
    TABLE_FILENAME,
    BundleProvenance,
    DatasetSpec,
    content_hash_of_table,
    is_dataset_directory,
    read_bundle,
    read_spec,
    read_table,
    sha256_of_file,
)
from remedi_data.bundle.bundle import normalize_table
from remedi_data.dataset.config import DatasetConfig
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.tasks import Split, split_codes_from_names
from remedi_data.logging_setup import setup_logging
from remedi_data.parallel import default_workers
from remedi_data.yaml_io import read_yaml, write_model, write_yaml

logger = logging.getLogger("migrate_dataset_identity")

#: The zarr group of the old layout that held the labels and the default split.
TASKS_GROUP = "tasks"
#: The files the migration rewrites; a dry run's shadow holds copies of them.
REWRITTEN_FILES = (DATASET_CONFIG_FILENAME, PROVENANCE_FILENAME)

Outcome = Literal["unchanged", "migrated", "would_migrate", "failed"]


class DatasetMigrationReport(BaseModel):
    """What migrating one dataset did, or would do on a dry run."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    outcome: Outcome
    changes: list[str] = Field(default_factory=list)
    #: Why it failed: the reason it could not be migrated, or what verify found.
    problems: list[str] = Field(default_factory=list)
    recorded_structures_sha256: str | None = None
    structures_sha256: str | None = None
    elapsed_seconds: float = 0.0


class MigrationReport(BaseModel):
    """Every dataset of one run."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    datasets: list[DatasetMigrationReport] = Field(default_factory=list)

    def count(self, outcome: Outcome) -> int:
        return sum(dataset.outcome == outcome for dataset in self.datasets)


class MigrationError(RuntimeError):
    """A dataset that cannot be migrated as it is; nothing was changed."""


@dataclass(frozen=True)
class LayoutChanges:
    """What the old layout of one dataset needs, found without changing it."""

    config: DatasetConfig
    removed_config_keys: list[str]
    tasks_arrays: list[str]


def dataset_directories(paths: list[Path]) -> list[Path]:
    """The dataset directories the given datasets and roots of datasets stand for.

    Raises:
        MigrationError: if a path is neither a dataset nor holds any.
    """
    datasets: list[Path] = []
    for path in map(Path, paths):
        if is_dataset_directory(path):
            datasets.append(path)
            continue
        children = (
            sorted(child for child in path.iterdir() if is_dataset_directory(child))
            if path.is_dir()
            else []
        )
        if not children:
            raise MigrationError(f"{path} is not a dataset and holds no datasets")
        datasets += children
    return datasets


def _tasks_problems(
    directory: Path, task_names: list[str], spec: DatasetSpec, table: pd.DataFrame
) -> list[str]:
    """Where the old ``tasks/`` group disagrees with the table it is dropped for."""
    group = zarr.open_group(
        store=LocalStore(directory / TASKS_GROUP, read_only=True), mode="r"
    )
    problems: list[str] = []
    if "targets_system" in group or "mask_system" in group:
        if task_names != spec.label_names():
            return [
                f"the zarr's task columns {task_names} are not the labels "
                f"{spec.label_names()} of dataset.yaml"
            ]
        values = table[task_names].to_numpy(dtype=np.float64, na_value=np.nan)
        present = ~np.isnan(values)
        stored_mask = np.asarray(group["mask_system"][:]).astype(bool)
        stored_targets = np.asarray(group["targets_system"][:], dtype=np.float64)
        if stored_mask.shape != values.shape or not np.array_equal(
            stored_mask, present
        ):
            problems.append("tasks/mask_system does not match the table's labels")
        elif not np.allclose(stored_targets[present], values[present], rtol=1e-6):
            problems.append("tasks/targets_system does not match the table's labels")
    if "split" in group:
        expected = (
            np.full(len(table), Split.unassigned.value, dtype=np.uint8)
            if spec.evaluation is None
            else split_codes_from_names(
                table[spec.evaluation.default_split].astype(str).tolist()
            )
        )
        if not np.array_equal(np.asarray(group["split"][:], dtype=np.uint8), expected):
            problems.append("tasks/split does not match the table's default split")
    return problems


def inspect_layout(directory: Path) -> LayoutChanges:
    """What the dataset's config and zarr need; reads only.

    Raises:
        MigrationError: if the config has no valid zarr geometry, the old task
            columns are not the spec's labels, or a value in ``tasks/`` differs
            from the table (deleting it would lose information).
    """
    raw = read_yaml(directory / DATASET_CONFIG_FILENAME) or {}
    known = DatasetConfig.model_fields
    config = DatasetConfig.model_validate(
        {key: value for key, value in raw.items() if key in known}
    )
    removed = sorted(key for key in raw if key not in known)
    tasks_directory = directory / TASKS_GROUP
    if not tasks_directory.exists():
        return LayoutChanges(config, removed, [])
    bundle = read_bundle(directory)
    tasks = raw.get("tasks") or {}
    task_names = [column["name"] for column in tasks.get("system_cols", [])]
    if tasks.get("atom_cols"):
        raise MigrationError("the zarr holds per-atom targets, which have no table")
    problems = _tasks_problems(directory, task_names, bundle.spec, bundle.table)
    if problems:
        raise MigrationError("; ".join(problems))
    arrays = sorted(
        path.name
        for path in tasks_directory.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    return LayoutChanges(config, removed, arrays)


def _remove_tasks_group(directory: Path) -> None:
    path = directory / TASKS_GROUP
    if path.is_symlink():  # a dry run's shadow: drop the link, not the target
        path.unlink()
    else:
        shutil.rmtree(path)


def _rewrite_provenance_output(
    directory: Path, keys: tuple[str, ...], value: str
) -> None:
    """Set one value under ``outputs`` in ``provenance.yaml``, nothing else.

    The document is edited as loaded and dumped with the writer's own options,
    so every other line stays as it was.
    """
    path = directory / PROVENANCE_FILENAME
    document = read_yaml(path)
    node = document["outputs"]
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    BundleProvenance.model_validate(document)
    write_yaml(path, document)


def rekey_table_content(directory: Path, report: DatasetMigrationReport) -> None:
    """Re-record the table's ``content_sha256`` if only its definition moved.

    Raises:
        MigrationError: if the recorded hash differs and the table's bytes are
            not the recorded ones either (a changed table, not a changed hash).
    """
    provenance = read_yaml(directory / PROVENANCE_FILENAME)
    record = provenance["outputs"][TABLE_FILENAME]
    actual = content_hash_of_table(
        normalize_table(read_table(directory), read_spec(directory))
    )
    if actual == record["content_sha256"]:
        return
    if sha256_of_file(directory / TABLE_FILENAME) != record["file_sha256"]:
        raise MigrationError(
            f"{TABLE_FILENAME} content_sha256 is {actual} but "
            f"{record['content_sha256']} is recorded, and its file_sha256 differs too"
        )
    _rewrite_provenance_output(directory, (TABLE_FILENAME, "content_sha256"), actual)
    report.changes.append(
        f"provenance.yaml: {TABLE_FILENAME} content_sha256 "
        f"{record['content_sha256']} -> {actual} (file bytes unchanged)"
    )


def migrate_layout(directory: Path, report: DatasetMigrationReport) -> None:
    """Apply the three changes to ``directory`` and record them in ``report``.

    Raises:
        MigrationError: if the dataset cannot be migrated as it is (see
            :func:`rekey_table_content` and :func:`inspect_layout`); at most a
            re-recorded table content hash was written then.
    """
    rekey_table_content(directory, report)
    changes = inspect_layout(directory)
    if changes.removed_config_keys:
        write_model(directory / DATASET_CONFIG_FILENAME, changes.config)
        report.changes.append(
            f"dataset_config.yaml: removed {', '.join(changes.removed_config_keys)}"
        )
    if (directory / TASKS_GROUP).exists():
        _remove_tasks_group(directory)
        report.changes.append(
            f"zarr: deleted the tasks group ({', '.join(changes.tasks_arrays)}), "
            "which matched the table"
        )
    provenance = read_yaml(directory / PROVENANCE_FILENAME)
    outputs = provenance.get("outputs") or {}
    report.recorded_structures_sha256 = outputs.get("structures_sha256")
    report.structures_sha256 = structures_sha256(
        MoleculeDataset.open_existing_dataset_from_dir(directory)
    )
    if report.structures_sha256 != report.recorded_structures_sha256:
        _rewrite_provenance_output(
            directory, ("structures_sha256",), report.structures_sha256
        )
        report.changes.append(
            "provenance.yaml: structures_sha256 "
            f"{report.recorded_structures_sha256} -> {report.structures_sha256}"
        )


def _shadow(directory: Path, root: Path) -> Path:
    """A directory under ``root`` that reads like ``directory``: every entry a
    symbolic link to the original, except copies of the files the migration
    rewrites."""
    shadow = root / directory.name
    shadow.mkdir()
    for entry in directory.iterdir():
        if entry.name in REWRITTEN_FILES:
            shutil.copy2(entry, shadow / entry.name)
        else:
            (shadow / entry.name).symlink_to(entry.resolve())
    return shadow


def migrate_dataset(
    directory: Path, *, dry_run: bool, workers: int, shadow_root: Path | None = None
) -> DatasetMigrationReport:
    """Migrate one dataset (or its shadow on a dry run) and verify the result.

    A dry run builds the shadow in a temporary directory under ``shadow_root``
    (default: the system's temporary directory). Failures are reported, not
    raised.
    """
    started = time.monotonic()
    directory = Path(directory)
    report = DatasetMigrationReport(dataset_path=directory, outcome="unchanged")
    try:
        if dry_run:
            with tempfile.TemporaryDirectory(
                prefix="dataset_migration_", dir=shadow_root
            ) as root:
                shadow = _shadow(directory, Path(root))
                migrate_layout(shadow, report)
                report.problems = verify_dataset(shadow, workers=workers).problems
        else:
            migrate_layout(directory, report)
            report.problems = verify_dataset(directory, workers=workers).problems
    except Exception as error:  # one dataset must not stop the others
        logger.exception("%s could not be migrated", directory)
        report.problems.append(f"{type(error).__name__}: {error}")
    if report.problems:
        report.outcome = "failed"
    elif report.changes:
        report.outcome = "would_migrate" if dry_run else "migrated"
    report.elapsed_seconds = time.monotonic() - started
    return report


def migrate(
    paths: list[Path],
    *,
    dry_run: bool,
    workers: int,
    shadow_root: Path | None = None,
) -> MigrationReport:
    """Migrate every dataset the paths stand for, one after the other.

    Raises:
        MigrationError: if a path is neither a dataset nor holds any.
    """
    report = MigrationReport(dry_run=dry_run)
    for directory in dataset_directories(paths):
        dataset_report = migrate_dataset(
            directory, dry_run=dry_run, workers=workers, shadow_root=shadow_root
        )
        logger.info(
            "%s: %s%s",
            directory,
            dataset_report.outcome,
            "".join(f"\n  {line}" for line in dataset_report.changes)
            + "".join(f"\n  problem: {line}" for line in dataset_report.problems),
        )
        report.datasets.append(dataset_report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="dataset directories, or directories whose children are datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="migrate and verify a shadow of each dataset; change nothing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="processes for verify's geometry checks (default: the job's CPUs)",
    )
    parser.add_argument(
        "--shadow-root",
        type=Path,
        help="where a dry run builds its shadows (default: the temporary directory)",
    )
    parser.add_argument("--report", type=Path, help="also write the report as yaml")
    arguments = parser.parse_args(argv)
    setup_logging()
    report = migrate(
        arguments.paths,
        dry_run=arguments.dry_run,
        workers=arguments.workers,
        shadow_root=arguments.shadow_root,
    )
    if arguments.report is not None:
        write_model(arguments.report, report)
    outcomes: list[Outcome] = ["unchanged", "migrated", "would_migrate", "failed"]
    print(", ".join(f"{report.count(outcome)} {outcome}" for outcome in outcomes))
    for dataset in report.datasets:
        if dataset.outcome == "failed":
            print(f"FAILED {dataset.dataset_path}: {'; '.join(dataset.problems)}")
    return 1 if report.count("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
