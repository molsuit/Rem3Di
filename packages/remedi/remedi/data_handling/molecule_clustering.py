"""Two-level Butina clustering of a dataset's molecules on the GPU (§5.2.3).

Butina at a coarse Tanimoto-distance cutoff over all molecules, then Butina at a
fine cutoff inside each coarse cluster. ``nvmolkit.clustering.fused_butina``
recomputes similarities on the fly, so memory is O(N). Fingerprints are ECFP of
``nonisomeric_smiles``, one per ``molecule_id``. The result is a cluster file
that ``remedi_data.selection`` draws from (``hierarchical_clusters``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from remedi_data.chemistry.fingerprints import packed_morgan_fingerprints
from remedi_data.selection import (
    ClusterRecord,
    SelectionError,
    dataset_identity,
    read_clusters,
    write_clusters,
)
from remedi_data.selection.config import NAME_PATTERN

logger = logging.getLogger(__name__)


class MoleculeClusteringConfig(BaseModel):
    """One clustering of one dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset: Path
    name: str = Field(pattern=NAME_PATTERN)
    radius: int = Field(default=2, gt=0)
    fingerprint_size: int = 2048
    #: Tanimoto distances (1 - similarity); molecules within the cutoff of a
    #: centroid join its cluster.
    coarse_cutoff: float = Field(gt=0.0, lt=1.0)
    fine_cutoff: float = Field(gt=0.0, lt=1.0)
    #: Processes for fingerprinting.
    workers: int = Field(default=1, gt=0)
    overwrite: bool = False

    @model_validator(mode="after")
    def _fine_inside_coarse(self) -> MoleculeClusteringConfig:
        if self.fine_cutoff >= self.coarse_cutoff:
            raise ValueError("fine_cutoff must be smaller than coarse_cutoff")
        if self.fingerprint_size not in (128, 256, 512, 1024, 2048):
            raise ValueError("fingerprint_size must be 128, 256, 512, 1024 or 2048")
        return self

    @property
    def fingerprint_description(self) -> str:
        return (
            f"Morgan radius {self.radius}, {self.fingerprint_size} bits, "
            "of nonisomeric_smiles"
        )


@dataclass(frozen=True)
class TwoLevelClusters:
    """Cluster ids per input row; fine ids are global and nest in coarse ones."""

    coarse: np.ndarray
    fine: np.ndarray
    coarse_seconds: float
    fine_seconds: float
    details: dict[str, float | int] = field(default_factory=dict)


def butina(fingerprints: torch.Tensor, cutoff: float) -> np.ndarray:
    """Butina cluster ids (0 = largest cluster) of packed fingerprints on the GPU.

    Raises:
        RuntimeError: if nvmolkit is not installed or CUDA is not available.
    """
    if len(fingerprints) == 1:
        return np.zeros(1, dtype=np.int64)
    try:
        from nvmolkit.clustering import fused_butina
    except ImportError as error:
        raise RuntimeError(
            "Butina clustering needs nvmolkit (uv sync --extra cuda) and a GPU"
        ) from error
    result = fused_butina(fingerprints, cutoff=cutoff, metric="tanimoto")
    torch.cuda.synchronize()
    return result.torch().cpu().numpy().astype(np.int64)


def two_level_butina(
    fingerprints: np.ndarray, coarse_cutoff: float, fine_cutoff: float
) -> TwoLevelClusters:
    """Butina at ``coarse_cutoff`` over all rows, then at ``fine_cutoff`` inside each cluster."""
    if not torch.cuda.is_available():
        raise RuntimeError("two-level Butina clustering needs a CUDA device")
    device_fingerprints = torch.from_numpy(fingerprints.view(np.int32)).cuda()
    started = time.monotonic()
    coarse = butina(device_fingerprints, coarse_cutoff)
    coarse_seconds = time.monotonic() - started
    logger.info(
        "coarse Butina (cutoff %.2f): %d rows -> %d clusters in %.1f s",
        coarse_cutoff,
        len(coarse),
        coarse.max() + 1,
        coarse_seconds,
    )

    started = time.monotonic()
    fine = np.empty_like(coarse)
    order = np.argsort(coarse, kind="stable")
    boundaries = np.flatnonzero(np.diff(coarse[order])) + 1
    next_fine_id = 0
    for members in np.split(order, boundaries):
        if len(members) == 1:
            fine[members] = next_fine_id
            next_fine_id += 1
            continue
        index = torch.from_numpy(members).cuda()
        local = butina(device_fingerprints[index], fine_cutoff)
        fine[members] = local + next_fine_id
        next_fine_id += int(local.max()) + 1
    fine_seconds = time.monotonic() - started
    logger.info(
        "fine Butina (cutoff %.2f): %d clusters in %.1f s",
        fine_cutoff,
        next_fine_id,
        fine_seconds,
    )
    return TwoLevelClusters(
        coarse=coarse,
        fine=fine,
        coarse_seconds=coarse_seconds,
        fine_seconds=fine_seconds,
        details=cluster_size_summary(coarse, "coarse")
        | cluster_size_summary(fine, "fine"),
    )


def cluster_size_summary(
    cluster_ids: np.ndarray, prefix: str
) -> dict[str, float | int]:
    """Counts and size quantiles of one level of a clustering."""
    sizes = np.bincount(cluster_ids)
    sizes = sizes[sizes > 0]
    return {
        f"{prefix}_clusters": len(sizes),
        f"{prefix}_singletons": int(np.sum(sizes == 1)),
        f"{prefix}_molecules_in_singletons": float(np.sum(sizes == 1) / sizes.sum()),
        f"{prefix}_median_size": float(np.median(sizes)),
        f"{prefix}_size_quantile_90": float(np.quantile(sizes, 0.9)),
        f"{prefix}_size_quantile_99": float(np.quantile(sizes, 0.99)),
        f"{prefix}_largest": int(sizes.max()),
    }


def molecule_smiles(dataset: Path) -> pl.DataFrame:
    """One ``nonisomeric_smiles`` per ``molecule_id``, sorted by ``molecule_id``.

    Raises:
        SelectionError: if the dataset has no SMILES.
    """
    table_path = Path(dataset) / "table.parquet"
    try:
        rows = pl.read_parquet(
            table_path, columns=["molecule_id", "nonisomeric_smiles"]
        )
    except pl.exceptions.ColumnNotFoundError as error:
        raise SelectionError(
            f"{dataset} has no nonisomeric_smiles, so it cannot be clustered"
        ) from error
    return rows.unique("molecule_id", keep="first").sort("molecule_id")


def cluster_molecules(config: MoleculeClusteringConfig) -> ClusterRecord:
    """Cluster a dataset's molecules and write ``clusters/<name>.{parquet,yaml}``.

    An existing cluster file made for the dataset as it is now is kept unless
    ``overwrite`` is set.
    """
    started = time.monotonic()
    identity = dataset_identity(config.dataset)
    if not config.overwrite:
        try:
            _, existing = read_clusters(
                config.dataset, config.name, identity.table_content_sha256
            )
        except SelectionError:
            existing = None
        if existing is not None:
            logger.info("%s/%s is up to date", identity.dataset_id, config.name)
            return existing

    molecules = molecule_smiles(config.dataset)
    fingerprint_started = time.monotonic()
    fingerprints = packed_morgan_fingerprints(
        molecules["nonisomeric_smiles"].to_list(),
        radius=config.radius,
        size=config.fingerprint_size,
        workers=config.workers,
    )
    fingerprint_seconds = time.monotonic() - fingerprint_started
    logger.info("%d fingerprints in %.1f s", len(fingerprints), fingerprint_seconds)
    clusters = two_level_butina(fingerprints, config.coarse_cutoff, config.fine_cutoff)
    record = ClusterRecord(
        name=config.name,
        dataset_id=identity.dataset_id,
        table_content_sha256=identity.table_content_sha256,
        fingerprint=config.fingerprint_description,
        method="two-level Butina, nvmolkit fused_butina (Tanimoto)",
        coarse_cutoff=config.coarse_cutoff,
        fine_cutoff=config.fine_cutoff,
        molecules=len(molecules),
        coarse_clusters=int(clusters.coarse.max()) + 1,
        fine_clusters=int(clusters.fine.max()) + 1,
        details={
            "fingerprint_seconds": fingerprint_seconds,
            "coarse_seconds": clusters.coarse_seconds,
            "fine_seconds": clusters.fine_seconds,
            "device": torch.cuda.get_device_name(),
            **clusters.details,
        },
        created_at=datetime.now(UTC),
        elapsed_seconds=time.monotonic() - started,
    )
    write_clusters(
        config.dataset,
        pl.DataFrame(
            {
                "molecule_id": molecules["molecule_id"],
                "coarse_cluster": clusters.coarse,
                "fine_cluster": clusters.fine,
            }
        ),
        record,
    )
    return record
