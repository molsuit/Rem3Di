"""The on-disk descriptor cache, keyed by data and descriptor identity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from remedi_data.bundle import structures_identity
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.config import DescriptorConfig


def descriptor_cache_path(
    config: DescriptorConfig,
    dataset: MoleculeDataset,
    cache_dir: Path,
    dataset_id: str,
) -> Path:
    """Where the descriptor matrix for one (dataset, descriptor) pair is cached.

    ``{dataset_id}__{name}__{structures}__{model}.npz`` (§7c). The two hashes
    are what make the key *correct* rather than merely conventional:

    ``structures``
        identifies the data — see
        :func:`remedi_data.bundle.structures_identity`. Keying on
        ``dataset_id`` alone served a matrix computed on an earlier ingest of
        the same endpoint, and the row-count mismatch only surfaced as an
        ``IndexError`` when the split mask was applied.
    ``model``
        identifies the descriptor — the checkpoint weights for a REM3DI model,
        the config itself for a fingerprint — so a renamed or retrained model
        cannot inherit another one's vectors.

    Twelve hex characters of each is 48 bits per hash; these are collision
    guards on a directory of at most a few hundred files, not signatures.
    """
    return Path(cache_dir) / (
        f"{dataset_id}__{config.name}"
        f"__{structures_identity(Path(dataset.path))[:12]}"
        f"__{config.identity()[:12]}.npz"
    )


def compute_and_cache(
    config: DescriptorConfig,
    dataset: MoleculeDataset,
    cache_dir: Path,
    dataset_id: str,
) -> np.ndarray:
    """Compute the descriptor matrix, caching it under :func:`descriptor_cache_path`.

    Raises:
        FileNotFoundError: if the dataset directory carries no
            ``provenance.yaml`` to identify its contents by.
    """
    cache_dir = Path(cache_dir)
    cache_path = descriptor_cache_path(config, dataset, cache_dir, dataset_id)
    if cache_path.exists():
        return np.asarray(np.load(cache_path)["X"])
    X = config.build().calculate(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X)
    return X
