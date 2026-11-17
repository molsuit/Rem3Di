"""Make, write and read selections (§5.2.3).

```
<dataset>/selections/<name>.parquet   structure_id (int64, sorted, unique)
<dataset>/selections/<name>.yaml      SelectionRecord (written last: the completion marker)
```

A selection records the dataset's table and structure hashes and the files its
steps read (pool datasets' table hashes, cluster-file hashes); reading it after
the dataset or one of those inputs changed fails, so a stale selection is never
used.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from remedi_data.chemistry.splits import murcko_scaffold
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.format import (
    is_dataset_directory,
    read_provenance,
    read_spec,
    read_table_polars,
    sha256_of_file,
)
from remedi_data.parallel import process_map
from remedi_data.selection.config import (
    ConformersPerMolecule,
    ExcludeOverlap,
    Filter,
    HierarchicalClusters,
    RandomMolecules,
    SelectionBatchConfig,
    SelectionConfig,
    SelectionError,
    SelectionInput,
    SelectionRecord,
    SelectionStep,
    StepCounts,
)
from remedi_data.selection.order import (
    cluster_paths,
    hierarchical_order,
    random_order,
    read_clusters,
)
from remedi_data.yaml_io import read_model, write_model

logger = logging.getLogger(__name__)

SELECTIONS_DIRECTORY = "selections"
ATOM_COUNT_COLUMN = "atom_count"
_SCAFFOLD_CHUNK = 20_000


@dataclass(frozen=True)
class DatasetIdentity:
    """The hashes a selection is tied to."""

    dataset_id: str
    table_content_sha256: str
    structures_sha256: str


def dataset_identity(dataset: Path) -> DatasetIdentity:
    """The dataset id and the hashes its provenance records.

    Raises:
        SelectionError: if the directory is not a dataset with recorded outputs.
    """
    dataset = Path(dataset)
    if not is_dataset_directory(dataset):
        raise SelectionError(f"{dataset} is not a dataset directory")
    outputs = read_provenance(dataset).outputs
    if outputs is None or outputs.structures_sha256 is None:
        raise SelectionError(f"{dataset} records no table and structure hashes")
    return DatasetIdentity(
        dataset_id=read_spec(dataset).dataset_id,
        table_content_sha256=outputs.table_parquet.content_sha256,
        structures_sha256=outputs.structures_sha256,
    )


def selection_paths(dataset: Path, name: str) -> tuple[Path, Path]:
    directory = Path(dataset) / SELECTIONS_DIRECTORY
    return directory / f"{name}.parquet", directory / f"{name}.yaml"


def _sha256_of_ids(structure_ids: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(structure_ids, dtype="<i8").tobytes()
    ).hexdigest()


def _molecule_count(rows: pl.DataFrame) -> int:
    return rows["molecule_id"].n_unique()


def _keep_molecules(rows: pl.DataFrame, molecule_ids: np.ndarray) -> pl.DataFrame:
    return rows.filter(pl.col("molecule_id").is_in(pl.Series(molecule_ids).implode()))


def _require_columns(rows: pl.DataFrame, step: SelectionStep, columns: Iterable[str]):
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise SelectionError(
            f"step {step.kind} needs columns {missing}, which the table does not have"
        )


def _scaffolds(smiles: list[str], workers: int) -> list[str | None]:
    return process_map(
        partial(murcko_scaffold, include_chirality=False),
        smiles,
        workers=workers,
        chunk_size=_SCAFFOLD_CHUNK,
    )


def _overlap_keys(smiles: list[str], match_on: str, workers: int) -> list[str]:
    """The key each SMILES is matched on: itself, or its scaffold (acyclic: itself)."""
    if match_on == "nonisomeric_smiles":
        return smiles
    keys = []
    for text, scaffold in zip(smiles, _scaffolds(smiles, workers), strict=True):
        if scaffold is None:
            raise SelectionError(f"cannot compute the Murcko scaffold of {text!r}")
        keys.append(f"scaffold:{scaffold}" if scaffold else f"smiles:{text}")
    return keys


def pool_datasets(paths: Iterable[Path]) -> list[Path]:
    """The dataset directories a list of datasets and dataset roots stands for.

    Raises:
        SelectionError: if a path is neither a dataset nor holds any.
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
            raise SelectionError(f"{path} is not a dataset and holds no datasets")
        datasets += children
    return datasets


def _pool_inputs(step: ExcludeOverlap) -> list[SelectionInput]:
    """Every pool dataset of ``step`` with its table's ``content_sha256``."""
    return [
        SelectionInput(
            path=dataset, content_sha256=dataset_identity(dataset).table_content_sha256
        )
        for dataset in pool_datasets(step.datasets)
    ]


def _cluster_input(dataset: Path, step: HierarchicalClusters) -> SelectionInput:
    """The cluster table of ``step`` with its file sha256.

    Raises:
        SelectionError: if the cluster table does not exist.
    """
    table_path, _ = cluster_paths(dataset, step.clusters)
    if not table_path.is_file():
        raise SelectionError(f"{dataset} has no cluster table {table_path.name}")
    return SelectionInput(path=table_path, content_sha256=sha256_of_file(table_path))


def step_inputs(dataset: Path, config: SelectionConfig) -> list[SelectionInput]:
    """The files the steps of ``config`` read besides ``dataset``, as they are now.

    The same list :func:`select_structures` records in ``SelectionRecord.inputs``,
    computed without running the steps: a selection whose recorded inputs
    differ from these was made from other data.

    Raises:
        SelectionError: if a pool dataset or a cluster table is missing.
    """
    inputs: list[SelectionInput] = []
    for step in config.steps:
        match step:
            case ExcludeOverlap():
                inputs += _pool_inputs(step)
            case HierarchicalClusters():
                inputs.append(_cluster_input(Path(dataset), step))
    return inputs


def _exclude_overlap(
    rows: pl.DataFrame, step: ExcludeOverlap, workers: int
) -> tuple[pl.DataFrame, list[SelectionInput]]:
    _require_columns(rows, step, ["nonisomeric_smiles"])
    inputs = _pool_inputs(step)
    pool_smiles: set[str] = set()
    for pool_input in inputs:
        dataset = pool_input.path
        if not read_spec(dataset).smiles:
            raise SelectionError(f"pool dataset {dataset} has no SMILES")
        pool_smiles.update(
            read_table_polars(dataset, columns=["nonisomeric_smiles"])[
                "nonisomeric_smiles"
            ].to_list()
        )
    pool_keys = set(_overlap_keys(sorted(pool_smiles), step.match_on, workers))
    corpus_smiles = rows["nonisomeric_smiles"].unique().to_list()
    corpus_keys = _overlap_keys(corpus_smiles, step.match_on, workers)
    overlapping = [
        text
        for text, key in zip(corpus_smiles, corpus_keys, strict=True)
        if key in pool_keys
    ]
    overlapping_molecules = rows.filter(
        pl.col("nonisomeric_smiles").is_in(
            pl.Series(overlapping, dtype=pl.String).implode()
        )
    )["molecule_id"].unique()
    logger.info(
        "exclude_overlap: %d pool keys from %d datasets; %d molecules overlap",
        len(pool_keys),
        len(inputs),
        len(overlapping_molecules),
    )
    return rows.filter(
        ~pl.col("molecule_id").is_in(overlapping_molecules.implode())
    ), inputs


def _conformers_per_molecule(
    rows: pl.DataFrame, step: ConformersPerMolecule
) -> pl.DataFrame:
    _require_columns(rows, step, [step.rank_by, step.group_by])
    kept = rows.filter(pl.col(step.rank_by).is_not_null())
    if step.minimum is not None:
        kept = kept.filter(pl.col(step.rank_by) >= step.minimum)
    if step.count is None:
        return kept
    return (
        kept.sort(
            [step.group_by, step.rank_by, "structure_id"],
            descending=[False, step.descending, False],
        )
        .filter(pl.int_range(pl.len()).over(step.group_by) < step.count)
        .sort("structure_id")
    )


def _filter(rows: pl.DataFrame, step: Filter) -> pl.DataFrame:
    try:
        return rows.filter(pl.sql_expr(step.expression))
    except Exception as error:
        raise SelectionError(
            f"filter {step.expression!r} failed on the table: {error}"
        ) from error


def _draw(rows: pl.DataFrame, order: np.ndarray, step) -> pl.DataFrame:
    count = step.molecules_to_draw(len(order))
    return _keep_molecules(rows, order[:count])


def _hierarchical_clusters(
    rows: pl.DataFrame,
    step: HierarchicalClusters,
    dataset: Path,
    identity: DatasetIdentity,
) -> tuple[pl.DataFrame, list[SelectionInput]]:
    assignments, _ = read_clusters(
        dataset, step.clusters, identity.table_content_sha256
    )
    present = pl.DataFrame({"molecule_id": rows["molecule_id"].unique()})
    joined = present.join(assignments, on="molecule_id", how="left")
    unclustered = joined.filter(pl.col("coarse_cluster").is_null()).height
    if unclustered:
        raise SelectionError(
            f"cluster file {step.clusters!r} has no cluster for {unclustered} molecules"
        )
    order = hierarchical_order(
        joined["molecule_id"].to_numpy(),
        joined["coarse_cluster"].to_numpy(),
        joined["fine_cluster"].to_numpy(),
        seed=step.seed,
        weighting=step.cluster_weighting,
    )
    return _draw(rows, order, step), [_cluster_input(dataset, step)]


def _apply_step(
    rows: pl.DataFrame,
    step: SelectionStep,
    dataset: Path,
    identity: DatasetIdentity,
    workers: int,
) -> tuple[pl.DataFrame, list[SelectionInput]]:
    match step:
        case RandomMolecules():
            order = random_order(rows["molecule_id"].unique().to_numpy(), step.seed)
            return _draw(rows, order, step), []
        case HierarchicalClusters():
            return _hierarchical_clusters(rows, step, dataset, identity)
        case ConformersPerMolecule():
            return _conformers_per_molecule(rows, step), []
        case ExcludeOverlap():
            return _exclude_overlap(rows, step, workers)
        case Filter():
            return _filter(rows, step), []


def _read_rows(dataset: Path) -> pl.DataFrame:
    """The table with an ``atom_count`` column from the zarr's atom pointer."""
    rows = read_table_polars(dataset)
    pointer = np.asarray(
        MoleculeDataset.open_existing_dataset_from_dir(Path(dataset)).ptr[:],
        dtype=np.int64,
    )
    if len(pointer) != rows.height + 1:
        raise SelectionError(
            f"{dataset}: the zarr holds {len(pointer) - 1} structures, "
            f"the table {rows.height}"
        )
    if ATOM_COUNT_COLUMN in rows.columns:
        raise SelectionError(f"{dataset}: the table already has {ATOM_COUNT_COLUMN!r}")
    atom_counts = np.diff(pointer)[rows["structure_id"].to_numpy()]
    return rows.with_columns(pl.Series(ATOM_COUNT_COLUMN, atom_counts))


def read_selection_record(dataset: Path, name: str) -> SelectionRecord:
    """Read ``selections/<name>.yaml``.

    Raises:
        SelectionError: if the selection does not exist or is incomplete.
    """
    table_path, record_path = selection_paths(dataset, name)
    if not record_path.is_file() or not table_path.is_file():
        raise SelectionError(f"{dataset} has no complete selection {name!r}")
    return read_model(record_path, SelectionRecord)


def _stale_reason(
    dataset: Path, record: SelectionRecord, identity: DatasetIdentity
) -> str | None:
    """Why ``record`` no longer describes ``dataset``; ``None`` if it still does.

    Compares the dataset's table and structure hashes and the step inputs (pool
    datasets' table hashes, cluster-file hashes) with what the record holds. A
    missing input counts as changed.
    """
    if (record.table_content_sha256, record.structures_sha256) != (
        identity.table_content_sha256,
        identity.structures_sha256,
    ):
        return (
            f"it was made from another build of {dataset} "
            "(table or structure hash differs)"
        )
    try:
        current = step_inputs(dataset, record.config)
    except (SelectionError, OSError) as error:
        return f"its inputs cannot be read any more ({error})"
    if current != record.inputs:
        changed = {
            str(entry.path)
            for entry in [*current, *record.inputs]
            if entry not in current or entry not in record.inputs
        }
        return f"its inputs changed since it was made: {sorted(changed)}"
    return None


def read_selection(dataset: Path, name: str) -> np.ndarray:
    """The sorted ``structure_id``s of a selection, checked against the dataset.

    Checks the dataset's hashes and the step inputs (pool datasets' table hashes
    and cluster-file hashes; re-hashing a cluster table costs one read of it).

    Raises:
        SelectionError: if the selection is missing, incomplete, or was made from
            other data than the dataset and its inputs now hold.
    """
    record = read_selection_record(dataset, name)
    reason = _stale_reason(Path(dataset), record, dataset_identity(dataset))
    if reason is not None:
        raise SelectionError(f"selection {name!r} is stale: {reason}; make it again")
    table_path, _ = selection_paths(dataset, name)
    structure_ids = (
        pl.read_parquet(table_path, columns=["structure_id"])["structure_id"]
        .to_numpy()
        .astype(np.int64)
    )
    if _sha256_of_ids(structure_ids) != record.selection_sha256:
        raise SelectionError(
            f"selection {name!r}: {table_path} does not match the hash its record holds"
        )
    return structure_ids


def _existing_record(dataset: Path, name: str) -> SelectionRecord | None:
    try:
        return read_selection_record(dataset, name)
    except SelectionError:
        return None


def select_structures(
    dataset: Path,
    config: SelectionConfig,
    *,
    workers: int = 1,
    overwrite: bool = False,
) -> SelectionRecord:
    """Make one selection of ``dataset`` and write it under ``selections/``.

    An existing selection with the same config, made from the dataset and the
    step inputs (pool datasets, cluster files) as they are now, is kept and its
    record returned.

    Raises:
        SelectionError: if a step cannot be applied, the selection is empty, or
            a different selection of that name exists and ``overwrite`` is off.
    """
    started = time.monotonic()
    dataset = Path(dataset)
    identity = dataset_identity(dataset)
    existing = _existing_record(dataset, config.name)
    if existing is not None:
        reason = (
            "it was made from another config"
            if existing.config != config
            else _stale_reason(dataset, existing, identity)
        )
        if reason is None:
            logger.info("%s/%s is up to date", identity.dataset_id, config.name)
            return existing
        if not overwrite:
            raise SelectionError(
                f"{dataset} already has a selection {config.name!r}, but {reason}; "
                "set overwrite to replace it"
            )

    rows = _read_rows(dataset)
    step_counts = []
    inputs: list[SelectionInput] = []
    for step in config.steps:
        structures_in, molecules_in = rows.height, _molecule_count(rows)
        rows, step_inputs = _apply_step(rows, step, dataset, identity, workers)
        inputs += step_inputs
        step_counts.append(
            StepCounts(
                kind=step.kind,
                structures_in=structures_in,
                structures_out=rows.height,
                molecules_in=molecules_in,
                molecules_out=_molecule_count(rows),
            )
        )
        logger.info(
            "%s/%s %s: %d -> %d structures",
            identity.dataset_id,
            config.name,
            step.kind,
            structures_in,
            rows.height,
        )
    if rows.height == 0:
        raise SelectionError(f"selection {config.name!r} of {dataset} is empty")

    structure_ids = np.sort(rows["structure_id"].to_numpy().astype(np.int64))
    record = SelectionRecord(
        name=config.name,
        dataset_id=identity.dataset_id,
        table_content_sha256=identity.table_content_sha256,
        structures_sha256=identity.structures_sha256,
        config=config,
        steps=step_counts,
        inputs=inputs,
        structures=len(structure_ids),
        molecules=_molecule_count(rows),
        atoms=int(rows[ATOM_COUNT_COLUMN].sum()),
        selection_sha256=_sha256_of_ids(structure_ids),
        created_at=datetime.now(UTC),
        elapsed_seconds=time.monotonic() - started,
    )
    table_path, record_path = selection_paths(dataset, config.name)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.unlink(missing_ok=True)
    temporary = table_path.with_suffix(f".parquet.{os.getpid()}.tmp")
    pl.DataFrame({"structure_id": structure_ids}).write_parquet(temporary)
    temporary.replace(table_path)
    write_model(record_path, record)
    return record


@dataclass(frozen=True)
class SelectionFailure:
    dataset: Path
    name: str
    error: str


def select_batch(config: SelectionBatchConfig) -> list[SelectionRecord]:
    """Make every selection of a batch config.

    Raises:
        SelectionError: listing every selection that failed, after trying all of
            them (``keep_going``) or at the first failure.
    """
    records = []
    failures: list[SelectionFailure] = []
    for corpus in config.corpora:
        for selection in corpus.selections:
            try:
                records.append(
                    select_structures(
                        corpus.dataset,
                        selection,
                        workers=config.workers,
                        overwrite=config.overwrite,
                    )
                )
            except (SelectionError, OSError) as error:
                if not config.keep_going:
                    raise
                logger.error("%s/%s failed: %s", corpus.dataset, selection.name, error)
                failures.append(
                    SelectionFailure(corpus.dataset, selection.name, str(error))
                )
    if failures:
        listing = "\n".join(f"  {f.dataset}/{f.name}: {f.error}" for f in failures)
        raise SelectionError(f"{len(failures)} selections failed:\n{listing}")
    return records
