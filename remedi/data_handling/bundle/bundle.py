"""Reading and writing a *prepared benchmark bundle* directory (§1).

A bundle is one directory::

    <benchmark_root>/<dataset_id>/
        benchmark.yaml       # BenchmarkSpec                     (required)
        table.parquet        # one row per structure             (required)
        structures.extxyz    # 3D geometries, row-aligned        (iff stage: conformers)
        provenance.yaml      # BundleProvenance                  (required)

Nothing else. ``write_bundle`` refuses to write an invalid bundle and
``read_bundle`` re-validates everything it reads, so a preparer that writes the
wrong columns fails loudly on the first read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from remedi.data_handling.bundle.provenance import (
    BundleOutputs,
    BundleProvenance,
    StructuresOutputRecord,
    TableOutputRecord,
)
from remedi.data_handling.bundle.spec import BenchmarkSpec
from remedi.data_handling.bundle.validate import (
    count_stereoisomer_straddling_constitutions,
    validate_bundle,
)

SPEC_FILENAME = "benchmark.yaml"
TABLE_FILENAME = "table.parquet"
STRUCTURES_FILENAME = "structures.extxyz"
PROVENANCE_FILENAME = "provenance.yaml"


class BundleValidationError(ValueError):
    """Raised when a bundle violates the format. Lists every problem found."""

    def __init__(self, directory: Path | None, problems: list[str]):
        self.directory = directory
        self.problems = list(problems)
        where = f" in {directory}" if directory is not None else ""
        super().__init__(
            f"invalid benchmark bundle{where}:\n  " + "\n  ".join(self.problems)
        )


@dataclass
class Bundle:
    """A prepared benchmark bundle held in memory."""

    spec: BenchmarkSpec
    table: pd.DataFrame
    structures: list[Atoms] | None
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


def normalize_table(table: pd.DataFrame, spec: BenchmarkSpec) -> pd.DataFrame:
    """Coerce the declared columns to the dtypes of §1.1, leaving extras alone.

    Applied on both sides of the disk round trip so that ``content_sha256`` is
    a property of the data and not of the parquet reader's dtype guesses.
    """
    normalized = table.copy()
    for column_name in ("structure_id", "stereoisomer_id", "molecule_id"):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype("int64")
    for column_name in ("isomeric_smiles", "nonisomeric_smiles", *spec.split_columns):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype("str")
    if "enantiomer_of" in normalized.columns:
        normalized["enantiomer_of"] = normalized["enantiomer_of"].astype("Int64")
    for task in spec.tasks:
        if task.name in normalized.columns:
            normalized[task.name] = normalized[task.name].astype("float64")
    ordered = [
        column_name
        for column_name in spec.expected_columns()
        if column_name in normalized.columns
    ]
    remaining = [
        column_name for column_name in normalized.columns if column_name not in ordered
    ]
    return normalized[ordered + remaining]


# ------------------------------------------------------------------ counting


def _fill_derived_provenance(
    provenance: BundleProvenance, spec: BenchmarkSpec, table: pd.DataFrame
) -> BundleProvenance:
    """Fill the parts of ``provenance.yaml`` that follow from the data itself."""
    filled = provenance.model_copy(deep=True)
    filled.dataset_id = spec.dataset_id
    filled.prepared_at = datetime.now(UTC)
    filled.counts.final_rows = len(table)
    filled.counts.per_split = {
        str(value): int(count)
        for value, count in table[spec.default_split].value_counts().items()
    }
    filled.counts.per_task_non_null = {
        task.name: int(table[task.name].notna().sum()) for task in spec.tasks
    }
    filled.counts.stereoisomer_straddling_constitutions = (
        count_stereoisomer_straddling_constitutions(table, spec.default_split)
    )
    return filled


# ------------------------------------------------------------------------ io


def write_bundle(bundle: Bundle, directory: Path) -> Path:
    """Validate ``bundle`` and write the four files. Returns ``directory``.

    Raises:
        BundleValidationError: if any invariant of §1.1 is violated. Nothing is
            written in that case.
    """
    directory = Path(directory)
    table = normalize_table(bundle.table, bundle.spec)
    to_write = Bundle(
        spec=bundle.spec,
        table=table,
        structures=bundle.structures,
        provenance=bundle.provenance,
    )
    problems = validate_bundle(to_write)
    if problems:
        raise BundleValidationError(directory, problems)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / SPEC_FILENAME).write_text(
        yaml.safe_dump(
            bundle.spec.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )
    )
    table.to_parquet(directory / TABLE_FILENAME, index=False)
    if bundle.structures is not None:
        ase_write(directory / STRUCTURES_FILENAME, bundle.structures, format="extxyz")

    provenance = _fill_derived_provenance(bundle.provenance, bundle.spec, table)
    outputs = BundleOutputs(
        table_parquet=TableOutputRecord(
            file_sha256=sha256_of_file(directory / TABLE_FILENAME),
            content_sha256=content_hash_of_table(table),
            rows=len(table),
        )
    )
    if bundle.structures is not None:
        outputs.structures_extxyz = StructuresOutputRecord(
            file_sha256=sha256_of_file(directory / STRUCTURES_FILENAME),
            frames=len(bundle.structures),
        )
    provenance.outputs = outputs
    (directory / PROVENANCE_FILENAME).write_text(
        yaml.safe_dump(
            provenance.model_dump(mode="json", by_alias=True, exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )
    )
    return directory


def read_bundle(directory: Path) -> Bundle:
    """Read and fully validate a bundle directory.

    Raises:
        FileNotFoundError: if one of the required files is missing.
        BundleValidationError: if ``format_version`` is not 1, if any invariant
            of §1.1 is violated, or if the recorded hashes do not match.
    """
    directory = Path(directory)
    for filename in (SPEC_FILENAME, TABLE_FILENAME, PROVENANCE_FILENAME):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"{directory} has no {filename}")

    raw_spec = yaml.safe_load((directory / SPEC_FILENAME).read_text())
    if not isinstance(raw_spec, dict):
        raise BundleValidationError(directory, [f"{SPEC_FILENAME} is not a mapping"])
    format_version = raw_spec.get("format_version")
    if format_version != 1:
        raise BundleValidationError(
            directory,
            [f"format_version {format_version!r} is not supported (this reader is 1)"],
        )
    spec = BenchmarkSpec.model_validate(raw_spec)
    provenance = BundleProvenance.model_validate(
        yaml.safe_load((directory / PROVENANCE_FILENAME).read_text())
    )

    table = normalize_table(pd.read_parquet(directory / TABLE_FILENAME), spec)
    structures_path = directory / STRUCTURES_FILENAME
    structures: list[Atoms] | None = None
    if structures_path.is_file():
        structures = list(ase_read(structures_path, index=":"))
    elif spec.stage == "conformers":
        raise FileNotFoundError(
            f"{directory} is a conformers-stage bundle but has no {STRUCTURES_FILENAME}"
        )

    bundle = Bundle(
        spec=spec, table=table, structures=structures, provenance=provenance
    )
    problems = validate_bundle(bundle)
    problems += _check_recorded_hashes(bundle, directory)
    if problems:
        raise BundleValidationError(directory, problems)
    return bundle


def _check_recorded_hashes(bundle: Bundle, directory: Path) -> list[str]:
    """Compare the table's content hash and the extxyz file hash to the record.

    ``sha256(table.parquet)`` is deliberately *not* enforced: it is a writer
    hash and moves with compression or row-group settings (§1.4).
    """
    problems: list[str] = []
    outputs = bundle.provenance.outputs
    if outputs is None:
        return [f"{PROVENANCE_FILENAME} has no outputs block"]
    recorded_content = outputs.table_parquet.content_sha256
    actual_content = content_hash_of_table(bundle.table)
    if recorded_content != actual_content:
        problems.append(
            f"{TABLE_FILENAME} content_sha256 is {actual_content} but "
            f"{PROVENANCE_FILENAME} records {recorded_content}"
        )
    if outputs.table_parquet.rows != len(bundle.table):
        problems.append(
            f"{PROVENANCE_FILENAME} records {outputs.table_parquet.rows} rows but "
            f"{TABLE_FILENAME} has {len(bundle.table)}"
        )
    structures_path = directory / STRUCTURES_FILENAME
    if structures_path.is_file():
        if outputs.structures_extxyz is None:
            problems.append(
                f"{STRUCTURES_FILENAME} exists but {PROVENANCE_FILENAME} does not "
                "record it"
            )
        else:
            actual_file = sha256_of_file(structures_path)
            if outputs.structures_extxyz.file_sha256 != actual_file:
                problems.append(
                    f"{STRUCTURES_FILENAME} file_sha256 is {actual_file} but "
                    f"{PROVENANCE_FILENAME} records "
                    f"{outputs.structures_extxyz.file_sha256}"
                )
    elif outputs.structures_extxyz is not None:
        problems.append(
            f"{PROVENANCE_FILENAME} records {STRUCTURES_FILENAME} but the file is "
            "absent"
        )
    return problems


def read_table(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``table.parquet`` alone, optionally only ``columns``, without validating.

    For consumers that want a few columns (ids, splits, a label) and do not
    need the geometry or the invariants. Use :func:`read_bundle` whenever the
    bundle is being *used* rather than inspected.

    Raises:
        FileNotFoundError: if the directory holds no ``table.parquet``.
    """
    directory = Path(directory)
    table_path = directory / TABLE_FILENAME
    if not table_path.is_file():
        raise FileNotFoundError(f"{directory} has no {TABLE_FILENAME}")
    return pd.read_parquet(table_path, columns=columns)


def discover_bundles(root: Path) -> list[Path]:
    """Every bundle directory under ``root``, sorted. A bundle has a benchmark.yaml."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path.parent for path in root.rglob(SPEC_FILENAME) if path.is_file())
