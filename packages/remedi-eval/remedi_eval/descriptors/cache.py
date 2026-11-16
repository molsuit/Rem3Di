"""The descriptor cache for descriptors ``remedi_eval`` computes itself (ECFP).

A cached matrix is an embeddings pair (:mod:`embeddings_file`) at
``<cache_root>/<dataset_id>/<name>.npz``, with the descriptor config's hash as
``descriptor_identity``. A pair whose ``structures_sha256`` or identity does
not match the request, or that does not read back intact, is recomputed and
overwritten (:func:`~remedi_eval.descriptors.embeddings_file.ensure_embeddings`,
under a per-pair file lock), so one name holds one variant at a time and a
stale matrix is never served.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.base import DescriptorCalculator
from remedi_eval.descriptors.embeddings_file import ensure_embeddings, producer_record


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
    return ensure_embeddings(
        cache_root,
        dataset,
        dataset_id=dataset_id,
        descriptor_name=config.name,
        descriptor_identity=config.identity(),
        compute=lambda: config.build().calculate(dataset),
        producer=producer_record("remedi-eval"),
    ).embedding
