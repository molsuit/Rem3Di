"""Precomputed embeddings: one npz plus one sidecar yaml per dataset and descriptor.

The entry point for every frozen model (``BENCHMARK_DATA_FORMAT.md`` §7.5)::

    <embeddings_root>/<dataset_id>/<descriptor_name>.npz
      structure_id   (n_structures,)    int64, every structure of the dataset, any order
      embedding      (n_structures, d)  float32
    <embeddings_root>/<dataset_id>/<descriptor_name>.yaml
      EmbeddingsMetadata, including the npz's sha256

The yaml records the npz's sha256, so an npz replaced or copied without its
yaml is refused. The writer removes the old yaml first, writes the npz under a
unique temporary name and renames it into place, and writes the yaml last: a
yaml on disk means a complete pair. :func:`ensure_embeddings` is the one "up to
date, else compute and write" path: the Rem3Di embedding writer and the ECFP
descriptor cache both go through it, under a per-pair file lock
(``.<descriptor_name>.lock`` beside the pair), so jobs sharing a root never
compute one pair twice or write it over each other.
"""

from __future__ import annotations

import fcntl
import importlib.metadata
import logging
import os
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from remedi_data.dataset import Dataset
from remedi_data.format import git_head_sha, sha256_of_file, structures_identity
from remedi_data.yaml_io import read_model, write_model

logger = logging.getLogger(__name__)

EMBEDDINGS_FORMAT_VERSION = 1


class EmbeddingsFileError(ValueError):
    """An embeddings file pair that is incomplete, inconsistent or for other data."""


class ProducerRecord(BaseModel):
    """Which code wrote an embeddings file."""

    model_config = ConfigDict(extra="forbid")

    package: str
    version: str
    git_sha: str | None = None


class EmbeddingsMetadata(BaseModel):
    """The sidecar yaml of one embeddings npz."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = EMBEDDINGS_FORMAT_VERSION
    dataset_id: str
    # structures_identity() of the dataset the embeddings were computed on.
    structures_sha256: str
    descriptor_name: str
    # What produced the numbers: a checkpoint hash, a config hash.
    descriptor_identity: str
    npz_sha256: str
    n_structures: int
    embedding_dimension: int
    producer: ProducerRecord


@dataclass(frozen=True)
class EmbeddingsFile:
    """One read and checked embeddings pair."""

    metadata: EmbeddingsMetadata
    structure_id: np.ndarray
    embedding: np.ndarray


def embeddings_paths(
    embeddings_root: Path, dataset_id: str, descriptor_name: str
) -> tuple[Path, Path]:
    """The ``(npz, yaml)`` paths of one dataset's embeddings under ``embeddings_root``."""
    directory = Path(embeddings_root) / dataset_id
    return directory / f"{descriptor_name}.npz", directory / f"{descriptor_name}.yaml"


def producer_record(package: str) -> ProducerRecord:
    """The installed version of ``package`` and, if run from a git checkout, its commit."""
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return ProducerRecord(
        package=package,
        version=version,
        git_sha=git_head_sha(Path(__file__).resolve().parent),
    )


def write_embeddings(
    embeddings_root: Path,
    *,
    dataset_id: str,
    descriptor_name: str,
    descriptor_identity: str,
    structures_sha256: str,
    structure_id: np.ndarray,
    embedding: np.ndarray,
    producer: ProducerRecord,
) -> EmbeddingsMetadata:
    """Write one embeddings pair, replacing any previous one.

    Raises:
        EmbeddingsFileError: if the arrays have the wrong shape, the structure
            ids repeat, or the embedding holds a non-finite value.
    """
    structure_id = np.asarray(structure_id, dtype=np.int64)
    embedding = np.asarray(embedding, dtype=np.float32)
    if structure_id.ndim != 1 or embedding.ndim != 2:
        raise EmbeddingsFileError(
            f"structure_id must be 1-D and embedding 2-D, got shapes "
            f"{structure_id.shape} and {embedding.shape}"
        )
    if embedding.shape[0] != structure_id.shape[0]:
        raise EmbeddingsFileError(
            f"{embedding.shape[0]} embedding rows for {structure_id.shape[0]} structure ids"
        )
    if np.unique(structure_id).size != structure_id.size:
        raise EmbeddingsFileError(f"{dataset_id}: structure ids are not unique")
    if not np.isfinite(embedding).all():
        bad_rows = int((~np.isfinite(embedding)).any(axis=1).sum())
        raise EmbeddingsFileError(
            f"{dataset_id}/{descriptor_name}: {bad_rows} embedding rows hold NaN or inf"
        )

    npz_path, yaml_path = embeddings_paths(embeddings_root, dataset_id, descriptor_name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.unlink(missing_ok=True)
    temporary_npz = _temporary_path(npz_path)
    try:
        np.savez_compressed(
            temporary_npz, structure_id=structure_id, embedding=embedding
        )
        temporary_npz.replace(npz_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
    metadata = EmbeddingsMetadata(
        dataset_id=dataset_id,
        structures_sha256=structures_sha256,
        descriptor_name=descriptor_name,
        descriptor_identity=descriptor_identity,
        npz_sha256=sha256_of_file(npz_path),
        n_structures=int(structure_id.size),
        embedding_dimension=int(embedding.shape[1]),
        producer=producer,
    )
    write_model(yaml_path, metadata)
    return metadata


def _temporary_path(path: Path) -> Path:
    """A name beside ``path`` no other writer uses, with the same suffix."""
    return path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")


def read_metadata(
    embeddings_root: Path, dataset_id: str, descriptor_name: str
) -> EmbeddingsMetadata | None:
    """The sidecar yaml, or ``None`` when there is none (no complete pair)."""
    _, yaml_path = embeddings_paths(embeddings_root, dataset_id, descriptor_name)
    if not yaml_path.exists():
        return None
    return read_model(yaml_path, EmbeddingsMetadata)


def read_embeddings(
    embeddings_root: Path, dataset_id: str, descriptor_name: str
) -> EmbeddingsFile:
    """Read one embeddings pair and check it against its own metadata.

    Raises:
        FileNotFoundError: if the yaml or the npz is missing.
        EmbeddingsFileError: if the npz's sha256, the arrays' shapes, or the
            names in the yaml disagree with what was asked for.
    """
    npz_path, yaml_path = embeddings_paths(embeddings_root, dataset_id, descriptor_name)
    metadata = read_metadata(embeddings_root, dataset_id, descriptor_name)
    if metadata is None:
        raise FileNotFoundError(
            f"no embeddings for {dataset_id}/{descriptor_name}: {yaml_path} is missing"
        )
    if not npz_path.exists():
        raise FileNotFoundError(f"{yaml_path} has no {npz_path.name} beside it")
    if (metadata.dataset_id, metadata.descriptor_name) != (dataset_id, descriptor_name):
        raise EmbeddingsFileError(
            f"{yaml_path} describes {metadata.dataset_id}/{metadata.descriptor_name}"
        )
    if sha256_of_file(npz_path) != metadata.npz_sha256:
        raise EmbeddingsFileError(
            f"{npz_path} does not match the npz_sha256 recorded in {yaml_path.name}; "
            "the npz was replaced or the pair was copied apart"
        )
    with np.load(npz_path) as arrays:
        structure_id = np.asarray(arrays["structure_id"], dtype=np.int64)
        embedding = np.asarray(arrays["embedding"])
    expected_shape = (metadata.n_structures, metadata.embedding_dimension)
    if embedding.shape != expected_shape or structure_id.shape != expected_shape[:1]:
        raise EmbeddingsFileError(
            f"{npz_path}: shapes {structure_id.shape} / {embedding.shape} disagree "
            f"with the metadata's {expected_shape}"
        )
    return EmbeddingsFile(
        metadata=metadata, structure_id=structure_id, embedding=embedding
    )


def embeddings_aligned_to(
    embeddings: EmbeddingsFile, dataset_dir: Path, dataset_structure_ids: np.ndarray
) -> np.ndarray:
    """The embedding matrix in the dataset's structure order.

    ``dataset_structure_ids`` is the dataset's ``structure_id`` column in structure
    order.

    Raises:
        EmbeddingsFileError: if the embeddings were computed on other data (a
            different ``structures_sha256``) or cover a different set of
            structures than the dataset holds.
    """
    dataset_structures = structures_identity(Path(dataset_dir))
    if embeddings.metadata.structures_sha256 != dataset_structures:
        raise EmbeddingsFileError(
            f"embeddings {embeddings.metadata.dataset_id}/"
            f"{embeddings.metadata.descriptor_name} were computed on structures "
            f"{embeddings.metadata.structures_sha256[:12]}, but {dataset_dir} holds "
            f"{dataset_structures[:12]}; recompute them for this build"
        )
    dataset_structure_ids = np.asarray(dataset_structure_ids, dtype=np.int64)
    order = np.argsort(embeddings.structure_id)
    sorted_ids = embeddings.structure_id[order]
    positions = np.searchsorted(sorted_ids, dataset_structure_ids)
    positions = np.clip(positions, 0, max(sorted_ids.size - 1, 0))
    if sorted_ids.size != dataset_structure_ids.size or not np.array_equal(
        sorted_ids[positions], dataset_structure_ids
    ):
        raise EmbeddingsFileError(
            f"embeddings {embeddings.metadata.dataset_id}/"
            f"{embeddings.metadata.descriptor_name} cover {sorted_ids.size} structures, "
            f"not the {dataset_structure_ids.size} structures of {dataset_dir}"
        )
    return embeddings.embedding[order[positions]]


@dataclass(frozen=True)
class EnsuredEmbeddings:
    """One dataset's embedding matrix in structure order, and whether it was computed now."""

    embedding: np.ndarray
    computed: bool


@contextmanager
def _pair_lock(
    embeddings_root: Path, dataset_id: str, descriptor_name: str
) -> Generator[None]:
    """Hold an exclusive ``flock`` on the pair's lock file.

    The lock file stays on disk: removing it would let a waiting job lock an
    unlinked file while a third one locks a new one.
    """
    npz_path, _ = embeddings_paths(embeddings_root, dataset_id, descriptor_name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(npz_path.with_name(f".{descriptor_name}.lock"), "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _up_to_date_embeddings(
    embeddings_root: Path,
    dataset_id: str,
    descriptor_name: str,
    descriptor_identity: str,
    structures_sha256: str,
    dataset_dir: Path,
    structure_ids: np.ndarray,
) -> np.ndarray | None:
    """The stored matrix in structure order if the pair matches the request, else ``None``.

    A pair for other structures or another identity, and a pair that cannot be
    read or does not match its own metadata, are reported and not used.
    """
    try:
        metadata = read_metadata(embeddings_root, dataset_id, descriptor_name)
        if metadata is None:
            return None
        if (metadata.structures_sha256, metadata.descriptor_identity) != (
            structures_sha256,
            descriptor_identity,
        ):
            logger.info(
                "recomputing %s/%s: the stored pair is for other data or settings",
                dataset_id,
                descriptor_name,
            )
            return None
        stored = read_embeddings(embeddings_root, dataset_id, descriptor_name)
        return embeddings_aligned_to(stored, dataset_dir, structure_ids)
    except (OSError, ValueError) as error:
        logger.warning(
            "recomputing %s/%s: the stored pair cannot be used (%s)",
            dataset_id,
            descriptor_name,
            error,
        )
        return None


def ensure_embeddings(
    embeddings_root: Path,
    dataset: Dataset,
    *,
    dataset_id: str,
    descriptor_name: str,
    descriptor_identity: str,
    compute: Callable[[], np.ndarray],
    producer: ProducerRecord,
) -> EnsuredEmbeddings:
    """The pair of ``dataset`` under ``embeddings_root``, computed only if needed.

    Reuses the stored pair when it was made from the dataset's current
    structures (``structures_sha256``) by ``descriptor_identity`` and reads back
    intact; otherwise calls ``compute`` (the matrix in structure order) and writes the
    pair. Both happen under the pair's file lock, so a second job asking for the
    same pair waits for the first and then reuses its result.

    Raises:
        FileNotFoundError: if the dataset has no ``provenance.yaml`` to identify
            its structures by.
        EmbeddingsFileError: if the computed matrix cannot be written (wrong
            shape, non-finite values).
    """
    dataset_dir = Path(dataset.path)
    structures_sha256 = structures_identity(dataset_dir)
    structure_ids = (
        dataset.table(["structure_id"])["structure_id"].to_numpy().astype(np.int64)
    )
    with _pair_lock(embeddings_root, dataset_id, descriptor_name):
        stored = _up_to_date_embeddings(
            embeddings_root,
            dataset_id,
            descriptor_name,
            descriptor_identity,
            structures_sha256,
            dataset_dir,
            structure_ids,
        )
        if stored is not None:
            logger.info("%s/%s is up to date", dataset_id, descriptor_name)
            return EnsuredEmbeddings(embedding=stored, computed=False)
        embedding = np.asarray(compute(), dtype=np.float32)
        write_embeddings(
            embeddings_root,
            dataset_id=dataset_id,
            descriptor_name=descriptor_name,
            descriptor_identity=descriptor_identity,
            structures_sha256=structures_sha256,
            structure_id=structure_ids,
            embedding=embedding,
            producer=producer,
        )
        return EnsuredEmbeddings(embedding=embedding, computed=True)
