"""Shared, lazily-computed eval resources — compute once, share across tasks.

The expensive input to descriptor evaluation is the **embedding matrix** a
model produces over a dataset: every benchmark cell of one dataset, across
learners, wants the same one. This module formalises the ad-hoc caching that
preceded it into a typed, memoised provider.

A :class:`ResourceSpec` is a small value object with a stable ``key()`` and a
``build(cache)`` that may itself pull other resources from the cache, so
composed specs dedupe. The :class:`ResourceCache` memoises by key for the
lifetime of one run; on-disk persistence is delegated to the builders that
already have it (:func:`compute_and_cache` for embeddings), so re-runs stay
cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from remedi.data_handling.bundle import structures_identity
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.benchmark.descriptors import (
    DescriptorConfig,
    compute_and_cache,
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
    """An ``(N, D)`` descriptor matrix over ``dataset``, disk-cached by content.

    ``dataset`` is the opened :class:`MoleculeDataset`. The in-memory key is
    the on-disk one: ``dataset_id``, the descriptor's ``name``, the hash of the
    *data* and the hash of the *model* (§7c, see
    :func:`remedi.evaluation.benchmark.descriptors.descriptor_cache_path`). The
    name alone is not an identity — two ingests of one endpoint, or two
    checkpoints under one label, would otherwise share a matrix.
    """

    dataset_id: str
    descriptor: DescriptorConfig
    dataset: MoleculeDataset
    cache_dir: Path
    # Memoised so that repeated ``cache.get(spec)`` calls do not re-read the
    # provenance yaml (and, for a REM3DI model, re-hash the checkpoint).
    _memoised_key: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def key(self) -> str:
        if self._memoised_key is None:
            self._memoised_key = (
                f"embedding::{self.dataset_id}::{self.descriptor.name}"
                f"::{structures_identity(Path(self.dataset.path))[:12]}"
                f"::{self.descriptor.identity()[:12]}"
            )
        return self._memoised_key

    def build(self, cache: ResourceCache) -> np.ndarray:
        del cache  # no dependencies
        return compute_and_cache(
            self.descriptor, self.dataset, self.cache_dir, self.dataset_id
        )
