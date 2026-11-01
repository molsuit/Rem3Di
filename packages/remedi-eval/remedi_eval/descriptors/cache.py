"""The descriptor cache for descriptors ``remedi_eval`` computes itself (ECFP).

A cached matrix is an embeddings pair (:mod:`embeddings_file`) at
``<cache_root>/<dataset_id>/<name>.npz``, with the descriptor config's hash as
``descriptor_identity``. A pair whose ``structures_sha256`` or identity does
not match the request is recomputed and overwritten, so one name holds one
variant at a time and a stale matrix is never served.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from remedi_data.bundle import structures_identity
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.base import DescriptorCalculator
from remedi_eval.descriptors.embeddings_file import (
    embeddings_aligned_to,
    producer_record,
    read_embeddings,
    read_metadata,
    write_embeddings,
)

logger = logging.getLogger(__name__)


class CachedDescriptor(Protocol):
    """A descriptor config the cache can compute and identify."""

    name: str

    def identity(self) -> str: ...

    def build(self) -> DescriptorCalculator: ...


def compute_and_cache(
    config: CachedDescriptor,
    dataset: MoleculeDataset,
    cache_root: Path,
    dataset_id: str,
) -> np.ndarray:
    """The descriptor matrix in zarr order, from the cache or freshly computed.

    Raises:
        FileNotFoundError: if the dataset directory carries no
            ``provenance.yaml`` to identify its contents by.
    """
    dataset_dir = Path(dataset.path)
    structures_sha256 = structures_identity(dataset_dir)
    descriptor_identity = config.identity()
    structure_ids = np.asarray(dataset.structure_ids[:], dtype=np.int64)
    metadata = read_metadata(cache_root, dataset_id, config.name)
    if (
        metadata is not None
        and metadata.structures_sha256 == structures_sha256
        and metadata.descriptor_identity == descriptor_identity
    ):
        cached = read_embeddings(cache_root, dataset_id, config.name)
        return embeddings_aligned_to(cached, dataset_dir, structure_ids)
    if metadata is not None:
        logger.info(
            "recomputing %s/%s: the cached matrix is for other data or settings",
            dataset_id,
            config.name,
        )
    matrix = np.asarray(config.build().calculate(dataset), dtype=np.float32)
    write_embeddings(
        cache_root,
        dataset_id=dataset_id,
        descriptor_name=config.name,
        descriptor_identity=descriptor_identity,
        structures_sha256=structures_sha256,
        structure_id=structure_ids,
        embedding=matrix,
        producer=producer_record("remedi-eval"),
    )
    return matrix
