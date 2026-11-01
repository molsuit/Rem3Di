"""Precomputed embeddings: one npz plus one sidecar yaml per dataset and descriptor.

The entry point for every frozen model (``BENCHMARK_DATA_FORMAT.md`` §7.5)::

    <embeddings_root>/<dataset_id>/<descriptor_name>.npz
      structure_id   (n_structures,)    int64, every structure of the dataset, any order
      embedding      (n_structures, d)  float32
    <embeddings_root>/<dataset_id>/<descriptor_name>.yaml
      EmbeddingsMetadata, including the npz's sha256

The yaml records the npz's sha256, so an npz replaced or copied without its
yaml is refused. The writer removes the old yaml first, writes the npz, and
writes the yaml last: a yaml on disk means a complete pair. The ECFP descriptor
cache uses the same files.
"""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic_yaml as pyd_yaml
from pydantic import BaseModel, ConfigDict

from remedi_data.bundle import sha256_of_file, structures_identity

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
    return ProducerRecord(package=package, version=version, git_sha=_git_sha())


def _git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


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
    temporary_path = npz_path.with_name(npz_path.stem + ".partial.npz")
    np.savez_compressed(temporary_path, structure_id=structure_id, embedding=embedding)
    temporary_path.replace(npz_path)
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
    pyd_yaml.to_yaml_file(yaml_path, metadata)
    return metadata


def read_metadata(
    embeddings_root: Path, dataset_id: str, descriptor_name: str
) -> EmbeddingsMetadata | None:
    """The sidecar yaml, or ``None`` when there is none (no complete pair)."""
    _, yaml_path = embeddings_paths(embeddings_root, dataset_id, descriptor_name)
    if not yaml_path.exists():
        return None
    return pyd_yaml.parse_yaml_file_as(EmbeddingsMetadata, yaml_path)


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

    ``dataset_structure_ids`` is the dataset's ``ids/structure_id`` in zarr
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
