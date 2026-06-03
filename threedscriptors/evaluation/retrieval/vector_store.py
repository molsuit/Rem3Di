"""The shared retrieval object: a :class:`VectorStore`.

A vector store is an ``(N, D)`` embedding matrix produced by a trained REM3DI
model over a :class:`MoleculeDataset`, kept aligned with the per-structure SMILES
and structure ids, plus a nearest-neighbor index over the rows. The index lives
behind the :class:`RetrievalIndex` interface so the exact-sklearn backend can be
swapped for an approximate / GPU backend (e.g. faiss) later by adding a config
variant to :data:`RetrievalIndexConfig` — none of the task code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from pydantic import BaseModel, Field

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.benchmark.descriptors import RemediConfig

# -- Nearest-neighbor index --------------------------------------------------


class RetrievalIndex(ABC):
    """Minimal nearest-neighbor index: ``fit`` once, ``query`` many times."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Index the ``(N, D)`` matrix."""

    @abstractmethod
    def query(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, indices)`` each shaped ``(len(Q), k)``.

        Distances are in the index's metric (cosine distance = ``1 - cos_sim``
        for the cosine backend); indices are rows into the fitted matrix,
        nearest first.
        """


class SklearnFlatIndex(RetrievalIndex):
    """Exact brute-force index backed by ``sklearn.neighbors.NearestNeighbors``.

    No new dependency (scikit-learn is already an ``eval`` extra) and exact
    results; fine up to a few hundred thousand vectors.
    """

    def __init__(self, metric: Literal["cosine", "euclidean"] = "cosine") -> None:
        self.metric = metric
        self._nn = None
        self._n = 0

    def fit(self, X: np.ndarray) -> None:
        from sklearn.neighbors import NearestNeighbors

        self._n = int(X.shape[0])
        self._nn = NearestNeighbors(metric=self.metric, algorithm="brute")
        self._nn.fit(np.ascontiguousarray(X, dtype=np.float32))

    def query(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._nn is None:
            raise RuntimeError("SklearnFlatIndex.query called before fit().")
        k = min(k, self._n)
        dist, idx = self._nn.kneighbors(
            np.ascontiguousarray(Q, dtype=np.float32), n_neighbors=k
        )
        return dist, idx


class SklearnIndexConfig(BaseModel):
    index_kind: Literal["sklearn_flat"] = "sklearn_flat"
    metric: Literal["cosine", "euclidean"] = "cosine"

    def build(self) -> SklearnFlatIndex:
        return SklearnFlatIndex(metric=self.metric)


# Discriminated union with a single member today; add a `FaissIndexConfig`
# variant here (and a matching RetrievalIndex impl above) to scale out without
# touching VectorStore or any task module.
RetrievalIndexConfig = Annotated[
    SklearnIndexConfig, Field(discriminator="index_kind")
]


# -- Embedder protocol (for embedding brand-new query molecules) -------------


@runtime_checkable
class AtomsEmbedder(Protocol):
    """Anything that can embed explicit ASE molecules into the store's space."""

    def embed_atoms(self, atoms: list[Atoms]) -> np.ndarray: ...


# -- The store ---------------------------------------------------------------


@dataclass
class VectorStore:
    """An embedding matrix plus aligned chemistry metadata and an NN index.

    ``X[i]``, ``smiles[i]`` and ``structure_ids[i]`` all refer to the same
    dataset structure. ``embedder`` is optional and only needed to query the
    store with molecules that are not already in it (see :meth:`embed_atoms`).
    """

    X: np.ndarray
    smiles: list[str]
    structure_ids: np.ndarray
    index: RetrievalIndex
    embedder: AtomsEmbedder | None = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    def query(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` neighbors of each row in ``vectors``: ``(distances, indices)``."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        return self.index.query(vectors, k)

    def neighbors(
        self, row_indices: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` neighbors of stored rows, excluding each row itself.

        Used by the Tanimoto task, where every query is already a store entry
        and the trivial self-match must be dropped.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64)
        dist, idx = self.query(self.X[row_indices], k + 1)
        out_idx = np.full((len(row_indices), k), -1, dtype=np.int64)
        out_dist = np.full((len(row_indices), k), np.nan, dtype=np.float64)
        for r, src in enumerate(row_indices):
            keep = idx[r] != src
            kept_i = idx[r][keep][:k]
            kept_d = dist[r][keep][:k]
            out_idx[r, : len(kept_i)] = kept_i
            out_dist[r, : len(kept_d)] = kept_d
        return out_dist, out_idx

    def embed_atoms(self, atoms: list[Atoms]) -> np.ndarray:
        """Embed brand-new molecules through the same model that built the store."""
        if self.embedder is None:
            raise RuntimeError(
                "VectorStore has no embedder; cannot embed new molecules. Build "
                "the store via build_vector_store so the model is attached."
            )
        return self.embedder.embed_atoms(atoms)


def build_vector_store(
    model_config: RemediConfig,
    dataset: MoleculeDataset,
    *,
    cache_dir: Path,
    index_config: RetrievalIndexConfig,
    dataset_id: str,
    max_structures: int | None = None,
) -> VectorStore:
    """Embed ``dataset`` with the trained model, cache it, and index it.

    The embedding matrix is cached to ``{cache_dir}/{dataset_id}__{name}.npz``
    keyed by the model config's ``name`` (delete the file to force recompute).
    The same :class:`RemediCalculator` is attached as the store's embedder so
    Task C can embed new query molecules without reloading the checkpoint.
    """
    calculator = model_config.build()
    cache_dir = Path(cache_dir)
    # The cap is part of the cache identity so a 5k smoke run can never be
    # served (or overwritten) by a full-dataset run under the same name.
    suffix = f"__n{max_structures}" if max_structures is not None else ""
    cache_path = cache_dir / f"{dataset_id}__{model_config.name}{suffix}.npz"
    if cache_path.exists():
        X = np.asarray(np.load(cache_path)["X"], dtype=np.float32)
    else:
        X = calculator.calculate(dataset, limit=max_structures).astype(np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X)

    # X is already limited to `max_structures` rows by calculate(); align the
    # per-structure metadata to it.
    n_rows = X.shape[0]
    smiles = dataset.get_smiles_per_structure()[:n_rows]
    structure_ids = np.asarray(dataset.structure_ids[:n_rows])

    if len(smiles) != X.shape[0]:
        raise ValueError(
            f"SMILES/embedding misalignment: {len(smiles)} SMILES vs "
            f"{X.shape[0]} embedding rows for dataset {dataset_id!r}."
        )

    index = index_config.build()
    index.fit(X)
    return VectorStore(
        X=X,
        smiles=smiles,
        structure_ids=structure_ids,
        index=index,
        embedder=calculator,
    )
