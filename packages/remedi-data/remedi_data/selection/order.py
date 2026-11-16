"""Seeded drawing orders over molecules, and the cluster files they read (§5.2.3).

A draw of *n* molecules takes the first *n* of an order, so draws of every size
with one seed are nested. Random keys are taken per id from a stream seeded by
``(seed, stream)`` alone, so a molecule's key does not depend on which other
molecules are present.

Cluster files are written by the clustering task in ``remedi`` (it needs a GPU)
and read here:

```
<dataset>/clusters/<name>.parquet   molecule_id, coarse_cluster, fine_cluster (int64)
<dataset>/clusters/<name>.yaml      ClusterRecord
```

Fine cluster ids are global, and every fine cluster lies inside one coarse one.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from remedi_data.selection.config import ClusterWeighting, SelectionError
from remedi_data.yaml_io import read_model, write_model

CLUSTERS_DIRECTORY = "clusters"
CLUSTER_COLUMNS = ("molecule_id", "coarse_cluster", "fine_cluster")

_MOLECULE_STREAM = 0
_FINE_CLUSTER_STREAM = 1
_COARSE_CLUSTER_STREAM = 2


class ClusterRecord(BaseModel):
    """``clusters/<name>.yaml``: how a two-level clustering was made."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_id: str
    #: The dataset the clustering is valid for.
    table_content_sha256: str
    #: What was clustered, e.g. ``"ECFP4 (radius 2, 2048 bits) of nonisomeric_smiles"``.
    fingerprint: str
    method: str
    #: Tanimoto distance cutoffs.
    coarse_cutoff: float
    fine_cutoff: float
    molecules: int
    coarse_clusters: int
    fine_clusters: int
    #: Free-form numbers worth keeping (timings per level, largest cluster).
    details: dict[str, float | int | str] = Field(default_factory=dict)
    created_at: datetime
    elapsed_seconds: float


def cluster_paths(dataset: Path, name: str) -> tuple[Path, Path]:
    directory = Path(dataset) / CLUSTERS_DIRECTORY
    return directory / f"{name}.parquet", directory / f"{name}.yaml"


def cluster_problems(assignments: pl.DataFrame) -> list[str]:
    """What is wrong with a cluster assignment table; empty if nothing."""
    missing = [column for column in CLUSTER_COLUMNS if column not in assignments]
    if missing:
        return [f"cluster file lacks columns {missing}"]
    problems = []
    if assignments.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        problems.append("cluster file has null entries")
    if assignments["molecule_id"].n_unique() != assignments.height:
        problems.append("a molecule_id appears more than once")
    coarse_per_fine = assignments.group_by("fine_cluster").agg(
        pl.col("coarse_cluster").n_unique()
    )
    if (coarse_per_fine["coarse_cluster"] > 1).any():
        problems.append("a fine cluster spans several coarse clusters")
    return problems


def write_clusters(
    dataset: Path, assignments: pl.DataFrame, record: ClusterRecord
) -> None:
    """Write a cluster file and its record; the yaml goes last as the completion marker.

    Raises:
        SelectionError: if the assignments break the cluster file's rules.
    """
    problems = cluster_problems(assignments)
    if problems:
        raise SelectionError(f"cluster file {record.name!r}: {'; '.join(problems)}")
    table_path, record_path = cluster_paths(dataset, record.name)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.unlink(missing_ok=True)
    temporary = table_path.with_suffix(f".parquet.{os.getpid()}.tmp")
    assignments.select(
        [pl.col(column).cast(pl.Int64) for column in CLUSTER_COLUMNS]
    ).sort("molecule_id").write_parquet(temporary)
    temporary.replace(table_path)
    write_model(record_path, record)


def read_clusters(
    dataset: Path, name: str, table_content_sha256: str
) -> tuple[pl.DataFrame, ClusterRecord]:
    """Read a cluster file, checking that it belongs to the dataset as it is now.

    Raises:
        SelectionError: if it is missing, broken or made for other data.
    """
    table_path, record_path = cluster_paths(dataset, name)
    if not record_path.is_file() or not table_path.is_file():
        raise SelectionError(
            f"{dataset} has no complete cluster file {name!r} "
            f"(expected {table_path} and {record_path})"
        )
    record = read_model(record_path, ClusterRecord)
    if record.table_content_sha256 != table_content_sha256:
        raise SelectionError(
            f"cluster file {name!r} was made for table {record.table_content_sha256[:12]}, "
            f"but {dataset} now holds {table_content_sha256[:12]}; recluster"
        )
    assignments = pl.read_parquet(table_path)
    problems = cluster_problems(assignments)
    if problems:
        raise SelectionError(f"cluster file {name!r}: {'; '.join(problems)}")
    return assignments, record


def keys_by_id(seed: int, stream: int, ids: np.ndarray) -> np.ndarray:
    """A uniform random key per id, fixed by ``(seed, stream, id)``."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.zeros(0, dtype=np.float64)
    if ids.min() < 0:
        raise SelectionError("ids must be non-negative")
    return np.random.default_rng([seed, stream]).random(int(ids.max()) + 1)[ids]


def random_order(molecule_ids: np.ndarray, seed: int) -> np.ndarray:
    """``molecule_ids`` (unique) in the seeded random order."""
    molecule_ids = np.asarray(molecule_ids, dtype=np.int64)
    keys = keys_by_id(seed, _MOLECULE_STREAM, molecule_ids)
    return molecule_ids[np.lexsort((molecule_ids, keys))]


def _rank_within_groups(groups: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Each element's 0-based position within its group when sorted by ``keys``."""
    order = np.lexsort((keys, groups))
    sorted_groups = groups[order]
    is_start = np.ones(len(order), dtype=bool)
    is_start[1:] = sorted_groups[1:] != sorted_groups[:-1]
    start_position = np.maximum.accumulate(np.where(is_start, np.arange(len(order)), 0))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order)) - start_position
    return ranks


def _turn_times(
    groups: np.ndarray,
    ranks: np.ndarray,
    offsets: np.ndarray,
    weighting: ClusterWeighting,
) -> np.ndarray:
    """When the element of rank ``r`` in its group comes up in a weighted round-robin.

    A group of weight *w* comes up at times ``(r + offset) / w``: every 1/*w*,
    starting at a random phase, so over any stretch of the order each group gets
    turns in proportion to its weight.
    """
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if weighting == "uniform":
        weights = np.ones(len(sizes))
    elif weighting == "square_root":
        weights = np.sqrt(sizes)
    else:
        weights = sizes.astype(np.float64)
    return (ranks + offsets) / weights[inverse]


def hierarchical_order(
    molecule_ids: np.ndarray,
    coarse_clusters: np.ndarray,
    fine_clusters: np.ndarray,
    seed: int,
    weighting: ClusterWeighting,
) -> np.ndarray:
    """``molecule_ids`` (unique) in the diversity-first drawing order.

    Inside a fine cluster the molecules come in random order; inside a coarse
    cluster its fine clusters take weighted turns; over the dataset the coarse
    clusters take weighted turns.
    """
    molecule_ids = np.asarray(molecule_ids, dtype=np.int64)
    coarse_clusters = np.asarray(coarse_clusters, dtype=np.int64)
    fine_clusters = np.asarray(fine_clusters, dtype=np.int64)
    molecule_keys = keys_by_id(seed, _MOLECULE_STREAM, molecule_ids)
    rank_in_fine = _rank_within_groups(fine_clusters, molecule_keys)
    time_in_coarse = _turn_times(
        fine_clusters,
        rank_in_fine,
        keys_by_id(seed, _FINE_CLUSTER_STREAM, fine_clusters),
        weighting,
    )
    rank_in_coarse = _rank_within_groups(coarse_clusters, time_in_coarse)
    time = _turn_times(
        coarse_clusters,
        rank_in_coarse,
        keys_by_id(seed, _COARSE_CLUSTER_STREAM, coarse_clusters),
        weighting,
    )
    return molecule_ids[np.lexsort((molecule_keys, time))]
