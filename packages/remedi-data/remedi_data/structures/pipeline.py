"""Sharded, restartable preparation of one structure source into one dataset.

``prepare_structures`` splits the source's records into shards of
``shard_size``. Each shard runs in a worker process, turns every record into a
kept frame or a counted drop (:func:`~remedi_data.structures.records.prepare_record`)
and writes ``shards/<n>.parquet`` (the table rows), ``shards/<n>.npz`` (the
frames) and, last, ``shards/<n>.yaml`` (its counts; the completion marker). A
rerun skips every shard whose marker exists, so a crash or a timeout loses at
most the shards in flight. The final step concatenates the shards, assigns the
ids (factorising only: the canonical and mirror SMILES were computed per
shard) and writes the dataset through
:func:`~remedi_data.build.write.write_dataset`, which re-checks every invariant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from remedi_data.build.write import ZarrLayout, write_dataset
from remedi_data.bundle import (
    BundleCounts,
    BundleProvenance,
    DatasetSpec,
    FileHash,
    PreparerRecord,
    SourceRecord,
    identity_from_canonical,
    read_provenance,
    sha256_of_file,
)
from remedi_data.bundle.provenance import git_head_sha
from remedi_data.dataset.packed_structures import PackedStructures
from remedi_data.parallel import default_workers
from remedi_data.structures.records import (
    DroppedRecord,
    PreparedFrame,
    prepare_record,
)
from remedi_data.structures.sources import StructureSource
from remedi_data.yaml_io import read_model, write_model

logger = logging.getLogger(__name__)

#: Bump whenever a change to the record processing (``records.py``) or a
#: source reader changes which frames, SMILES or values a shard holds: the
#: fingerprint includes it, so shards and datasets from older code are refused
#: (shards) or rebuilt (datasets) instead of silently reused.
PIPELINE_VERSION = 3

PLAN_FILENAME = "plan.yaml"
SHARD_DIRECTORY = "shards"
RAW_HASHES_FILENAME = "raw_sha256.yaml"


class StructurePreparationSettings(BaseModel):
    """How a source is processed; none of this changes which data is written."""

    model_config = ConfigDict(extra="forbid")

    shard_size: int = Field(default=100_000, ge=1)
    workers: int = Field(default_factory=default_workers, ge=1)
    zarr: ZarrLayout = Field(default_factory=ZarrLayout)


class ShardSummary(BaseModel):
    """``shards/<n>.yaml``: what one shard read, kept and dropped."""

    model_config = ConfigDict(extra="forbid")

    index: int
    start: int
    stop: int
    records_read: int
    structures_kept: int
    dropped: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float


class PreparationPlan(BaseModel):
    """``plan.yaml``: which source, raw files and shard size the shards here belong to."""

    model_config = ConfigDict(extra="forbid")

    source_fingerprint: str
    #: file name -> sha256 of every raw file the shards were read from.
    raw_files: dict[str, str]
    shard_size: int
    record_count: int


class PreparationSummary(BaseModel):
    """The result of one source, as recorded in the run directory."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_path: Path
    skipped: bool = False
    records_read: int = 0
    structures_written: int = 0
    dropped: dict[str, int] = Field(default_factory=dict)
    shards: int = 0
    shards_reused: int = 0
    elapsed_seconds: float = 0.0


class ShardFailures(RuntimeError):
    """Raised after every shard ran when some of them failed; the others are kept."""


def source_fingerprint(source: StructureSource) -> str:
    """sha256 of the source config, its type and :data:`PIPELINE_VERSION`.

    Shards from another config or pipeline version are refused. The raw files'
    contents are checked separately, against ``plan.yaml`` and the dataset's
    ``provenance.yaml`` (:func:`_raw_file_hashes`).
    """
    document = {
        "pipeline_version": PIPELINE_VERSION,
        "type": type(source).__name__,
        **source.model_dump(mode="json"),
    }
    document.pop("description", None)
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _shard_stem(work_directory: Path, index: int) -> Path:
    return work_directory / SHARD_DIRECTORY / f"{index:05d}"


def _shard_table(
    source: StructureSource, frames: Sequence[PreparedFrame]
) -> pd.DataFrame:
    columns: dict[str, object] = {
        "source_key": pd.array([frame.source_key for frame in frames], dtype="str")
    }
    if source.smiles_filter is not None:
        for name in ("isomeric_smiles", "nonisomeric_smiles", "mirror_smiles"):
            columns[name] = pd.array(
                [getattr(frame, name) for frame in frames], dtype="str"
            )
    columns["total_charge"] = np.array(
        [frame.total_charge for frame in frames], dtype=np.float64
    )
    columns["multiplicity"] = np.array(
        [frame.multiplicity for frame in frames], dtype=np.float64
    )
    for label in source.labels():
        columns[label.name] = np.array(
            [frame.labels.get(label.name, np.nan) for frame in frames],
            dtype=np.float64,
        )
    for name, dtype in source.extra_columns().items():
        columns[name] = pd.array(
            [frame.extras.get(name) for frame in frames], dtype=dtype
        )
    return pd.DataFrame(columns)


def run_shard(
    source: StructureSource,
    index: int,
    start: int,
    stop: int,
    work_directory: Path,
) -> ShardSummary:
    """Read records ``start:stop``, keep or drop each, write the three shard files."""
    started = time.monotonic()
    limits = source.geometry_limits
    allowed = limits.allowed_atomic_numbers()
    frames: list[PreparedFrame] = []
    dropped: dict[str, int] = {}
    records_read = 0
    for record in source.read_records(start, stop, work_directory):
        records_read += 1
        if isinstance(record, DroppedRecord):
            outcome: PreparedFrame | str = record.reason
        else:
            outcome = prepare_record(
                record,
                smiles_filter=source.smiles_filter,
                limits=limits,
                allowed_atomic_numbers=allowed,
            )
        if isinstance(outcome, str):
            dropped[outcome] = dropped.get(outcome, 0) + 1
        else:
            frames.append(outcome)
    stem = _shard_stem(work_directory, index)
    stem.parent.mkdir(parents=True, exist_ok=True)
    partial_table = stem.with_suffix(".parquet.partial")
    _shard_table(source, frames).to_parquet(partial_table, index=False)
    partial_table.replace(stem.with_suffix(".parquet"))
    partial_frames = stem.with_name(f"{stem.name}.partial.npz")
    PackedStructures.from_frames(
        [(frame.atomic_numbers, frame.positions) for frame in frames]
    ).save_npz(partial_frames)
    partial_frames.replace(stem.with_suffix(".npz"))
    summary = ShardSummary(
        index=index,
        start=start,
        stop=stop,
        records_read=records_read,
        structures_kept=len(frames),
        dropped=dict(sorted(dropped.items())),
        elapsed_seconds=time.monotonic() - started,
    )
    write_model(stem.with_suffix(".yaml"), summary)
    return summary


def _read_shard_summary(work_directory: Path, index: int) -> ShardSummary | None:
    path = _shard_stem(work_directory, index).with_suffix(".yaml")
    if not path.exists():
        return None
    return read_model(path, ShardSummary)


def _changed_raw_files(
    recorded: dict[str, str], current: dict[str, FileHash]
) -> list[str]:
    """Names of the raw files whose sha256 differs, or that appear on one side only."""
    names = set(recorded) | set(current)
    return sorted(
        name
        for name in names
        if name not in recorded
        or name not in current
        or recorded[name] != current[name].sha256
    )


def _plan(
    source: StructureSource,
    settings: StructurePreparationSettings,
    work_directory: Path,
) -> PreparationPlan:
    """Load ``plan.yaml``, or write it; refuse shards made from another config
    or from other raw files."""
    fingerprint = source_fingerprint(source)
    raw_files = _raw_file_hashes(source, work_directory)
    path = work_directory / PLAN_FILENAME
    if path.exists():
        plan = read_model(path, PreparationPlan)
        if (
            plan.source_fingerprint != fingerprint
            or plan.shard_size != settings.shard_size
        ):
            raise ValueError(
                f"{work_directory} holds shards of another source config or shard "
                "size; delete it to start over"
            )
        changed = _changed_raw_files(plan.raw_files, raw_files)
        if changed:
            raise ValueError(
                f"{work_directory} holds shards read from other raw files "
                f"({', '.join(changed)} changed); delete it to start over"
            )
        return plan
    record_count = source.record_count(work_directory)
    if source.limit is not None:
        record_count = min(record_count, source.limit)
    plan = PreparationPlan(
        source_fingerprint=fingerprint,
        raw_files={name: entry.sha256 for name, entry in raw_files.items()},
        shard_size=settings.shard_size,
        record_count=record_count,
    )
    work_directory.mkdir(parents=True, exist_ok=True)
    write_model(path, plan)
    return plan


def run_shards(
    source: StructureSource,
    settings: StructurePreparationSettings,
    work_directory: Path,
) -> tuple[list[ShardSummary], int]:
    """Run every shard without a completion marker; returns all summaries and how
    many were reused.

    Raises:
        ShardFailures: after the pool drained, if any shard failed. Finished
            shards stay on disk for the rerun.
    """
    plan = _plan(source, settings, work_directory)
    bounds = [
        (index, start, min(start + plan.shard_size, plan.record_count))
        for index, start in enumerate(range(0, plan.record_count, plan.shard_size))
    ]
    summaries: dict[int, ShardSummary] = {}
    missing: list[tuple[int, int, int]] = []
    for index, start, stop in bounds:
        summary = _read_shard_summary(work_directory, index)
        if summary is None:
            missing.append((index, start, stop))
        else:
            summaries[index] = summary
    reused = len(summaries)
    logger.info(
        "%s: %d records in %d shards, %d already done",
        source.dataset_id,
        plan.record_count,
        len(bounds),
        reused,
    )
    failures: list[str] = []
    if missing and settings.workers > 1:
        context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(
            max_workers=min(settings.workers, len(missing)), mp_context=context
        ) as pool:
            futures: dict[Future[ShardSummary], int] = {
                pool.submit(
                    run_shard, source, index, start, stop, work_directory
                ): index
                for index, start, stop in missing
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    summaries[index] = future.result()
                    logger.info(
                        "%s: shard %d done (%d/%d)",
                        source.dataset_id,
                        index,
                        len(summaries),
                        len(bounds),
                    )
                except Exception as error:  # one shard must not sink the others
                    logger.exception("%s: shard %d failed", source.dataset_id, index)
                    failures.append(f"shard {index}: {error!r}")
    else:
        for index, start, stop in missing:
            try:
                summaries[index] = run_shard(source, index, start, stop, work_directory)
            except Exception as error:
                logger.exception("%s: shard %d failed", source.dataset_id, index)
                failures.append(f"shard {index}: {error!r}")
    if failures:
        raise ShardFailures(
            f"{source.dataset_id}: {len(failures)} of {len(bounds)} shards failed; "
            "rerun to retry only those:\n  " + "\n  ".join(sorted(failures))
        )
    return [summaries[index] for index, _, _ in bounds], reused


def _raw_file_hashes(
    source: StructureSource, work_directory: Path
) -> dict[str, FileHash]:
    """sha256 of every raw file, cached by path, size and modification time.

    The cache lives in ``work_directory``, which is created if needed.
    """
    work_directory.mkdir(parents=True, exist_ok=True)
    cache_path = work_directory / RAW_HASHES_FILENAME
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    hashes: dict[str, FileHash] = {}
    for path in source.raw_files():
        status = path.stat()
        key = str(path.resolve())
        entry = cache.get(key)
        if (
            entry is None
            or entry["size"] != status.st_size
            or entry["mtime"] != status.st_mtime
        ):
            entry = {
                "size": status.st_size,
                "mtime": status.st_mtime,
                "sha256": sha256_of_file(path),
            }
            cache[key] = entry
        hashes[path.name] = FileHash(sha256=str(entry["sha256"]))
    cache_path.write_text(json.dumps(cache, indent=1))
    return hashes


def dataset_spec(source: StructureSource) -> DatasetSpec:
    """The ``dataset.yaml`` a source's dataset gets."""
    return DatasetSpec(
        dataset_id=source.dataset_id,
        description=source.description,
        smiles=source.smiles_filter is not None,
        geometry_origin="source",
        labels=source.labels(),
        extra_columns=["source_key", *source.extra_columns()],
        source_kind=source.kind,
    )


def assemble_table(
    source: StructureSource, shard_tables: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    """The dataset table from the shard tables: ids, then the declared columns."""
    spec = dataset_spec(source)
    combined = pd.concat(shard_tables, ignore_index=True)
    table = pd.DataFrame({"structure_id": np.arange(len(combined), dtype=np.int64)})
    if spec.smiles:
        identity = identity_from_canonical(
            combined["isomeric_smiles"].tolist(),
            combined["nonisomeric_smiles"].tolist(),
            combined["mirror_smiles"].tolist(),
        ).to_frame()
        for column in identity.columns:
            table[column] = identity[column].to_numpy()
        table["enantiomer_of"] = identity["enantiomer_of"].array
    else:
        identifier = pd.factorize(combined["source_key"].astype(str))[0].astype(
            np.int64
        )
        table["stereoisomer_id"] = identifier
        table["molecule_id"] = identifier
        table["enantiomer_of"] = pd.array([pd.NA] * len(combined), dtype="Int64")
    for column in spec.expected_columns():
        if column not in table.columns:
            table[column] = combined[column].to_numpy()
    return table[spec.expected_columns()]


def prepare_structures(
    source: StructureSource,
    settings: StructurePreparationSettings,
    output_directory: Path,
    work_directory: Path,
    *,
    overwrite: bool = False,
) -> PreparationSummary:
    """Prepare one source into the dataset at ``output_directory``.

    Skips the work when ``output_directory`` already holds a dataset made from
    the same source config and the same raw files (their sha256 against the
    dataset's ``provenance.yaml``), unless ``overwrite``.

    Raises:
        ShardFailures: if a shard failed (rerun to retry it).
        BundleValidationError: if the assembled dataset breaks an invariant.
        ValueError: if ``work_directory`` holds shards of another config or of
            other raw files.
    """
    started = time.monotonic()
    output_directory = Path(output_directory)
    work_directory = Path(work_directory)
    marker = output_directory / "source_fingerprint.txt"
    fingerprint = source_fingerprint(source)
    raw_files = _raw_file_hashes(source, work_directory)
    if (
        not overwrite
        and marker.exists()
        and marker.read_text().strip() == fingerprint
        and (output_directory / "provenance.yaml").exists()
        and not _changed_raw_files(
            {
                name: entry.sha256
                for name, entry in read_provenance(
                    output_directory
                ).source.files.items()
            },
            raw_files,
        )
    ):
        logger.info("%s: already prepared from this config", source.dataset_id)
        return PreparationSummary(
            dataset_id=source.dataset_id,
            dataset_path=output_directory,
            skipped=True,
        )
    summaries, reused = run_shards(source, settings, work_directory)
    shard_tables = [
        pd.read_parquet(
            _shard_stem(work_directory, summary.index).with_suffix(".parquet")
        )
        for summary in summaries
    ]
    structures = PackedStructures.concatenate(
        [
            PackedStructures.load_npz(
                _shard_stem(work_directory, summary.index).with_suffix(".npz")
            )
            for summary in summaries
        ]
    )
    table = assemble_table(source, shard_tables)
    dropped: dict[str, int] = {}
    for summary in summaries:
        for reason, count in summary.dropped.items():
            dropped[reason] = dropped.get(reason, 0) + count
    dropped = dict(sorted(dropped.items()))
    records_read = sum(summary.records_read for summary in summaries)
    # Source records (a GEOM molecule, an SDF entry, a tmQM complex), not the
    # records read: a GEOM molecule yields one record per conformer kept, or
    # one dropped record when its pickle is missing.
    source_molecules = sum(summary.stop - summary.start for summary in summaries)
    provenance = BundleProvenance(
        dataset_id=source.dataset_id,
        preparer=PreparerRecord(
            repo="Rem3Di",
            script="packages/remedi-data/remedi_data/structures/pipeline.py",
            git_sha=git_head_sha(Path(__file__).parent),
        ),
        source=SourceRecord(
            files=raw_files,
            geometry=source.geometry_description(),
        ),
        geometry_limits=source.geometry_limits,
        smiles_filter=source.smiles_filter,
        counts=BundleCounts(source_molecules=source_molecules, dropped=dropped),
    )
    logger.info(
        "%s: writing %d structures (%d records read, dropped %s)",
        source.dataset_id,
        len(table),
        records_read,
        dropped,
    )
    write_dataset(
        dataset_spec(source),
        table,
        structures,
        provenance,
        output_directory,
        layout=settings.zarr,
        workers=settings.workers,
    )
    marker.write_text(fingerprint + "\n")
    return PreparationSummary(
        dataset_id=source.dataset_id,
        dataset_path=output_directory,
        records_read=records_read,
        structures_written=len(table),
        dropped=dropped,
        shards=len(summaries),
        shards_reused=reused,
        elapsed_seconds=time.monotonic() - started,
    )
