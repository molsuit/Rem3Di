"""Shared, lazily-computed eval resources — compute once, share across tasks.

The expensive inputs to descriptor evaluation are the **embedding matrix** a
model produces over a dataset and the **kNN index** built on top of it. Several
tasks in one run want the same ones (the two retrieval sub-tasks share an
embedding + index; a descriptor-analysis task reuses the embedding). This
module formalises the ad-hoc ``DescriptorAnalysisContext.cache`` dict into a
typed, memoised provider.

A :class:`ResourceSpec` is a small value object with a stable ``key()`` and a
``build(cache)`` that may itself pull other resources from the cache (so an
``IndexSpec`` composes the ``EmbeddingSpec`` it indexes, and they dedupe). The
:class:`ResourceCache` memoises by key for the lifetime of one run; on-disk
persistence is delegated to the builders that already have it
(:func:`compute_and_cache` for embeddings), so re-runs stay cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.benchmark.descriptors import (
    DescriptorConfig,
    compute_and_cache,
)
from remedi.evaluation.retrieval.vector_store import (
    RetrievalIndex,
    RetrievalIndexConfig,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ResourceSpec(Protocol):
    """A memoisable resource: a stable identity plus a build recipe."""

    def key(self) -> str:
        """Stable identity; two specs with the same key share one build."""

    def build(self, cache: ResourceCache) -> Any:
        """Produce the resource. May pull dependencies from ``cache``."""


class ResourceCache:
    """In-process memoisation of resources by ``spec.key()``.

    One cache per run. ``get`` builds a resource at most once; builders that
    have a disk form (embeddings) persist it themselves so a *fresh* run reuses
    prior on-disk artifacts too.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._build_counts: dict[str, int] = {}

    def get(self, spec: ResourceSpec) -> Any:
        k = spec.key()
        if k not in self._store:
            logger.info("building resource %s", k)
            self._store[k] = spec.build(self)
            self._build_counts[k] = self._build_counts.get(k, 0) + 1
        return self._store[k]

    def build_count(self, spec: ResourceSpec) -> int:
        """How many times ``spec`` was actually built (0 or 1). For tests."""
        return self._build_counts.get(spec.key(), 0)


@dataclass
class EmbeddingSpec:
    """An ``(N, D)`` descriptor matrix over ``dataset``, disk-cached by name.

    ``dataset`` is the opened :class:`MoleculeDataset`; ``dataset_id`` and the
    descriptor's ``name`` form the on-disk cache key (reusing
    :func:`compute_and_cache`'s ``{dataset_id}__{name}.npz`` convention).
    """

    dataset_id: str
    descriptor: DescriptorConfig
    dataset: MoleculeDataset
    cache_dir: Path

    def key(self) -> str:
        return f"embedding::{self.dataset_id}::{self.descriptor.name}"

    def build(self, cache: ResourceCache) -> np.ndarray:
        del cache  # no dependencies
        return compute_and_cache(
            self.descriptor, self.dataset, self.cache_dir, self.dataset_id
        )


@dataclass
class IndexSpec:
    """A fitted nearest-neighbour index over an :class:`EmbeddingSpec` matrix."""

    embedding: EmbeddingSpec
    index_config: RetrievalIndexConfig

    def key(self) -> str:
        return f"index::{self.embedding.key()}::{self.index_config.index_kind}"

    def build(self, cache: ResourceCache) -> RetrievalIndex:
        X = np.ascontiguousarray(cache.get(self.embedding), dtype=np.float32)
        index = self.index_config.build()
        index.fit(X)
        return index


@dataclass
class FingerprintSpec:
    """An ECFP fingerprint matrix over a dataset's SMILES, disk-cached as npz.

    The reference descriptor for the Tanimoto retrieval baseline / any task that
    needs fingerprints alongside the learned embedding.
    """

    dataset_id: str
    dataset: MoleculeDataset
    cache_dir: Path
    radius: int = 2
    n_bits: int = 2048
    # Extra options forwarded to the molfeat featurizer kind string.
    kind: str = "ecfp"

    def key(self) -> str:
        return (
            f"fingerprint::{self.dataset_id}::{self.kind}{self.n_bits}_r{self.radius}"
        )

    def _cache_path(self) -> Path:
        return Path(self.cache_dir) / f"{self.dataset_id}__{self.kind}{self.n_bits}.npz"

    def build(self, cache: ResourceCache) -> np.ndarray:
        del cache
        path = self._cache_path()
        if path.exists():
            return np.asarray(np.load(path)["X"], dtype=np.float32)
        from molfeat.trans.fp import FPVecTransformer

        feat = FPVecTransformer(kind=self.kind, length=self.n_bits, radius=self.radius)
        X = np.asarray(feat(self.dataset.get_smiles_per_structure()), dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=X)
        return X


@dataclass
class _CallableSpec:
    """Adapter wrapping a plain ``(key, thunk)`` as a :class:`ResourceSpec`.

    For one-off in-memory resources that don't warrant their own spec class.
    """

    _key: str
    _thunk: Any = field(repr=False)

    def key(self) -> str:
        return self._key

    def build(self, cache: ResourceCache) -> Any:
        del cache
        return self._thunk()
