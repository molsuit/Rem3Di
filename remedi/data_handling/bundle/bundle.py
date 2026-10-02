"""Reading and writing the three files of a bundle or a dataset (§10.1).

A **bundle** directory is exactly::

    <bundle_root>/<dataset_id>/
        dataset.yaml       # DatasetSpec (geometry_origin unset)
        table.parquet      # one row per stereoisomer
        provenance.yaml    # BundleProvenance

A **dataset** directory is a zarr (written by ``dataset_build.write_dataset``)
with the same three files beside it, its table holding one row per structure.
``write_bundle`` refuses to write an invalid bundle and ``read_bundle`` (which
reads either kind) re-validates the table and its recorded hash on every open.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from remedi.data_handling.bundle.provenance import (
    BundleOutputs,
    BundleProvenance,
    TableOutputRecord,
)
from remedi.data_handling.bundle.spec import DatasetSpec
from remedi.data_handling.bundle.validate import (
    count_stereoisomer_straddling_constitutions,
    validate_table,
)

SPEC_FILENAME = "dataset.yaml"
TABLE_FILENAME = "table.parquet"
PROVENANCE_FILENAME = "provenance.yaml"
#: Written by ``MoleculeDataset.create_empty_dataset``; its presence is what
#: distinguishes a dataset directory from a bundle directory.
DATASET_CONFIG_FILENAME = "dataset_config.yaml"
#: The format version this reader understands.
FORMAT_VERSION = 1


class BundleValidationError(ValueError):
    """Raised when a bundle or dataset violates the format. Lists every problem."""

    def __init__(self, directory: Path | None, problems: list[str]):
        self.directory = directory
        self.problems = list(problems)
        where = f" in {directory}" if directory is not None else ""
        super().__init__(
            f"invalid bundle or dataset{where}:\n  " + "\n  ".join(self.problems)
        )


@dataclass
class Bundle:
    """The three files of a bundle or dataset, held in memory."""

    spec: DatasetSpec
    table: pd.DataFrame
    provenance: BundleProvenance


# ------------------------------------------------------------------- hashing


def sha256_of_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_dtype_label(series: pd.Series) -> str:
    """A writer- and pandas-version-independent name for a column's dtype."""
    dtype = series.dtype
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return "string"
    if str(dtype) == "Int64":
        return "Int64"
    return str(dtype)


def content_hash_of_table(table: pd.DataFrame) -> str:
    """sha256 over the *logical* content: column names, dtypes and row hashes.

    This is the comparability key of §1.4. Unlike ``sha256(table.parquet)`` it
    does not move when the parquet writer's compression or row-group size
    changes. String columns are hashed as plain objects so that the hash does
    not depend on which pandas string backend is in use.
    """
    digest = hashlib.sha256()
    hashable = pd.DataFrame(index=table.index)
    for column_name in table.columns:
        series = table[column_name]
        digest.update(f"{column_name}:{_canonical_dtype_label(series)}\n".encode())
        if _canonical_dtype_label(series) == "string":
            hashable[column_name] = series.astype(object)
        else:
            hashable[column_name] = series
    digest.update(
        pd.util.hash_pandas_object(hashable, index=False).to_numpy().tobytes()
    )
    return digest.hexdigest()


# ------------------------------------------------------------- normalisation


def normalize_table(table: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Coerce the declared columns to their format dtypes, in declared order.

    Applied on both sides of the disk round trip so that ``content_sha256`` is
    a property of the data and not of the parquet reader's dtype guesses.
    """
    normalized = table.copy()
    for column_name in ("structure_id", "stereoisomer_id", "molecule_id"):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype("int64")
    for column_name in ("isomeric_smiles", "nonisomeric_smiles", *spec.split_columns()):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype("str")
    if "enantiomer_of" in normalized.columns:
        normalized["enantiomer_of"] = normalized["enantiomer_of"].astype("Int64")
    for column_name in ("total_charge", "multiplicity", *spec.label_names()):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype("float64")
    ordered = [
        column_name
        for column_name in spec.expected_columns()
        if column_name in normalized.columns
    ]
    remaining = [
        column_name for column_name in normalized.columns if column_name not in ordered
    ]
    return normalized[ordered + remaining].reset_index(drop=True)


def fill_derived_provenance(
    provenance: BundleProvenance, spec: DatasetSpec, table: pd.DataFrame
) -> BundleProvenance:
    """A copy of ``provenance`` with everything that follows from the data filled in.

    ``outputs`` is left to the writer, which knows the file it wrote.
    """
    if provenance.dataset_id != spec.dataset_id:
        raise BundleValidationError(
            None,
            [
                f"provenance.dataset_id {provenance.dataset_id!r} != "
                f"spec.dataset_id {spec.dataset_id!r}"
            ],
        )
    filled = provenance.model_copy(deep=True)
    filled.prepared_at = datetime.now(UTC)
    filled.counts.final_rows = len(table)
    filled.counts.per_label_non_null = {
        label: int(table[label].notna().sum()) for label in spec.label_names()
    }
    if spec.evaluation is not None:
        default_split = spec.evaluation.default_split
        filled.counts.per_split = {
            str(value): int(count)
            for value, count in table[default_split].value_counts().items()
        }
        filled.counts.stereoisomer_straddling_constitutions = (
            count_stereoisomer_straddling_constitutions(table, default_split)
        )
    return filled


# ------------------------------------------------------------------------ io


def _dump_yaml(model, path: Path, **dump_options) -> None:
    path.write_text(
        yaml.safe_dump(
            model.model_dump(mode="json", exclude_none=True, **dump_options),
            sort_keys=False,
            allow_unicode=True,
        )
    )


def write_table_files(
    directory: Path,
    spec: DatasetSpec,
    table: pd.DataFrame,
    provenance: BundleProvenance,
    *,
    structures_sha256: str | None = None,
) -> BundleProvenance:
    """Write ``dataset.yaml``, ``table.parquet`` and ``provenance.yaml`` atomically.

    ``table`` must already be normalized and validated. Each file is written
    under a temporary name in ``directory`` and renamed into place (atomic per
    file on one filesystem), ``provenance.yaml``, which carries the hashes of
    the others, last; a concurrent reader never sees a half-written table.
    Returns the provenance as written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = fill_derived_provenance(provenance, spec, table)
    staged: list[tuple[Path, Path]] = []
    try:
        spec_staging = directory / f".{SPEC_FILENAME}.partial"
        _dump_yaml(spec, spec_staging)
        staged.append((spec_staging, directory / SPEC_FILENAME))
        table_staging = directory / f".{TABLE_FILENAME}.partial"
        table.to_parquet(table_staging, index=False)
        staged.append((table_staging, directory / TABLE_FILENAME))
        written.outputs = BundleOutputs(
            table_parquet=TableOutputRecord(
                file_sha256=sha256_of_file(table_staging),
                content_sha256=content_hash_of_table(table),
                rows=len(table),
            ),
            structures_sha256=structures_sha256,
        )
        provenance_staging = directory / f".{PROVENANCE_FILENAME}.partial"
        _dump_yaml(written, provenance_staging, by_alias=True)
        staged.append((provenance_staging, directory / PROVENANCE_FILENAME))
        for staging_path, final_path in staged:
            os.replace(staging_path, final_path)
    finally:
        for staging_path, _ in staged:
            staging_path.unlink(missing_ok=True)
    return written


def write_bundle(bundle: Bundle, directory: Path) -> BundleProvenance:
    """Validate ``bundle`` and write its three files. Returns the written provenance.

    Raises:
        BundleValidationError: if the spec declares structures (that is a
            dataset, written by ``dataset_build.write_dataset``) or any table
            invariant is violated. Nothing is written in that case.
    """
    directory = Path(directory)
    if bundle.spec.has_structures:
        raise BundleValidationError(
            directory,
            ["a bundle has no structures; leave geometry_origin unset"],
        )
    table = normalize_table(bundle.table, bundle.spec)
    problems = validate_table(bundle.spec, table)
    if problems:
        raise BundleValidationError(directory, problems)
    return write_table_files(directory, bundle.spec, table, bundle.provenance)


def read_bundle(directory: Path) -> Bundle:
    """Read and validate the three files of a bundle or dataset directory.

    Raises:
        FileNotFoundError: if one of the three files is missing.
        BundleValidationError: if ``format_version`` is not supported, a table
            invariant is violated, or the recorded table hash does not match.
    """
    directory = Path(directory)
    spec = read_spec(directory)
    provenance = read_provenance(directory)
    table = normalize_table(read_table(directory), spec)
    problems = validate_table(spec, table)
    problems += _check_recorded_table_hash(table, provenance)
    if problems:
        raise BundleValidationError(directory, problems)
    return Bundle(spec=spec, table=table, provenance=provenance)


def _check_recorded_table_hash(
    table: pd.DataFrame, provenance: BundleProvenance
) -> list[str]:
    """Compare the table's content hash and row count to the record.

    ``sha256(table.parquet)`` is deliberately *not* enforced: it is a writer
    hash and moves with compression or row-group settings (§1.4).
    """
    outputs = provenance.outputs
    if outputs is None:
        return [f"{PROVENANCE_FILENAME} has no outputs block"]
    problems: list[str] = []
    actual_content = content_hash_of_table(table)
    if outputs.table_parquet.content_sha256 != actual_content:
        problems.append(
            f"{TABLE_FILENAME} content_sha256 is {actual_content} but "
            f"{PROVENANCE_FILENAME} records {outputs.table_parquet.content_sha256}"
        )
    if outputs.table_parquet.rows != len(table):
        problems.append(
            f"{PROVENANCE_FILENAME} records {outputs.table_parquet.rows} rows but "
            f"{TABLE_FILENAME} has {len(table)}"
        )
    return problems


def read_table(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``table.parquet`` alone, optionally only ``columns``, without validating.

    Raises:
        FileNotFoundError: if the directory holds no ``table.parquet``.
    """
    table_path = Path(directory) / TABLE_FILENAME
    if not table_path.is_file():
        raise FileNotFoundError(f"{directory} has no {TABLE_FILENAME}")
    return pd.read_parquet(table_path, columns=columns)


def read_spec(directory: Path) -> DatasetSpec:
    """Read ``dataset.yaml`` alone.

    Raises:
        FileNotFoundError: if the directory holds no ``dataset.yaml``.
        BundleValidationError: if it is not a mapping, or declares a
            ``format_version`` this reader does not support.
    """
    directory = Path(directory)
    spec_path = directory / SPEC_FILENAME
    if not spec_path.is_file():
        raise FileNotFoundError(f"{directory} has no {SPEC_FILENAME}")
    raw_spec = yaml.safe_load(spec_path.read_text())
    if not isinstance(raw_spec, dict):
        raise BundleValidationError(directory, [f"{SPEC_FILENAME} is not a mapping"])
    if raw_spec.get("format_version") != FORMAT_VERSION:
        raise BundleValidationError(
            directory,
            [
                f"format_version {raw_spec.get('format_version')!r} is not supported "
                f"(this reader is {FORMAT_VERSION})"
            ],
        )
    return DatasetSpec.model_validate(raw_spec)


def read_provenance(directory: Path) -> BundleProvenance:
    """Read ``provenance.yaml`` alone.

    Raises:
        FileNotFoundError: if the directory holds no ``provenance.yaml``.
    """
    provenance_path = Path(directory) / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise FileNotFoundError(f"{directory} has no {PROVENANCE_FILENAME}")
    return BundleProvenance.model_validate(yaml.safe_load(provenance_path.read_text()))


def structures_identity(directory: Path) -> str:
    """The hash identifying the *data* anything derived from a directory saw (§7c).

    For a dataset that is ``structures_sha256``, the hash of the zarr's
    structures; for a bundle, the table's ``content_sha256``. A descriptor cache
    key must carry it: two builds of one ``dataset_id`` that dropped a different
    number of rows are different data under the same name.

    Raises:
        FileNotFoundError: if there is no ``provenance.yaml`` to read.
        BundleValidationError: if the provenance records no outputs at all.
    """
    directory = Path(directory)
    outputs = read_provenance(directory).outputs
    if outputs is None:
        raise BundleValidationError(
            directory,
            [
                f"{PROVENANCE_FILENAME} records no outputs, so the data this "
                "directory holds has no content hash to key a cache on"
            ],
        )
    if outputs.structures_sha256 is not None:
        return outputs.structures_sha256
    return outputs.table_parquet.content_sha256


def is_dataset_directory(directory: Path) -> bool:
    """A dataset directory holds a zarr (``dataset_config.yaml``) and ``dataset.yaml``."""
    directory = Path(directory)
    return (directory / SPEC_FILENAME).is_file() and (
        directory / DATASET_CONFIG_FILENAME
    ).is_file()


def discover_bundles(root: Path) -> list[Path]:
    """Every bundle directory under ``root`` (recursively), sorted."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        path.parent
        for path in root.rglob(SPEC_FILENAME)
        if path.is_file() and not is_dataset_directory(path.parent)
    )


def discover_datasets(root: Path) -> list[tuple[Path, DatasetSpec]]:
    """Every dataset directly under ``root``, with its spec, sorted by path.

    Anything else under ``root`` (run files, a bundle, a zarr without a spec)
    is skipped, so one directory can hold a whole panel.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return [
        (directory, read_spec(directory))
        for directory in sorted(path for path in root.iterdir() if path.is_dir())
        if is_dataset_directory(directory)
    ]
