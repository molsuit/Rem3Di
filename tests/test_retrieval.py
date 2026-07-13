"""Tests for the vector-retrieval eval.

These exercise the index, the Tanimoto math, and nearest-molecule lookup on a
*synthetic* store (embeddings fabricated from fingerprints), so they need no
trained model and no GPU. The model-dependent paths (embedding a dataset /
new-SMILES queries) are covered by the integration script, not here.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit.DataStructs import ConvertToNumpyArray

from remedi.evaluation.retrieval.config import (
    NearestMoleculeTaskConfig,
    TanimotoSimilarityTaskConfig,
)
from remedi.evaluation.retrieval.fingerprints import (
    morgan_fingerprints,
    tanimoto,
)
from remedi.evaluation.retrieval.nearest_molecule import (
    run_nearest_molecule,
    smiles_to_atoms,
)
from remedi.evaluation.retrieval.tanimoto_similarity import (
    run_tanimoto_similarity,
)
from remedi.evaluation.retrieval.vector_store import (
    SklearnIndexConfig,
    VectorStore,
)

# A small set with clear internal structure (alkanes / alcohols / benzenes /
# acids / amines) so embedding-space neighbors should track fingerprint
# neighbors when the embedding *is* the fingerprint.
_SMILES = [
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "Oc1ccccc1",
    "CC(=O)O",
    "CCC(=O)O",
    "CCCC(=O)O",
    "CCN",
    "CCCN",
]
_N_BITS = 512


def _fingerprint_store(metric: str = "cosine") -> VectorStore:
    fps = morgan_fingerprints(_SMILES, radius=2, n_bits=_N_BITS)
    X = np.zeros((len(fps), _N_BITS), dtype=np.float32)
    for i, fp in enumerate(fps):
        assert fp is not None
        ConvertToNumpyArray(fp, X[i])
    index = SklearnIndexConfig(metric=metric).build()
    index.fit(X)
    return VectorStore(
        X=X,
        smiles=list(_SMILES),
        structure_ids=np.arange(len(_SMILES)),
        index=index,
    )


def test_index_returns_self_first_for_exact_match():
    store = _fingerprint_store()
    dist, idx = store.query(store.X[3][None, :], k=1)
    assert idx[0, 0] == 3
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_neighbors_excludes_self():
    store = _fingerprint_store()
    rows = np.array([0, 7, 11])
    _, nbr = store.neighbors(rows, k=3)
    assert nbr.shape == (3, 3)
    for r, src in enumerate(rows):
        assert src not in set(nbr[r].tolist())


def test_tanimoto_helpers():
    fps = morgan_fingerprints(["CCO", "CCO", "not_a_smiles"], n_bits=_N_BITS)
    assert fps[2] is None
    assert tanimoto(fps[0], fps[1]) == pytest.approx(1.0)


def test_tanimoto_similarity_task_shows_enrichment(tmp_path):
    store = _fingerprint_store()
    cfg = TanimotoSimilarityTaskConfig(
        k=3, n_query_sample=len(_SMILES), n_global_pairs=400, fp_n_bits=_N_BITS
    )
    res = run_tanimoto_similarity(store, cfg, tmp_path)

    # Neighbors in (fingerprint-derived) embedding space are more similar than
    # random pairs, embedding similarity tracks Tanimoto, and the embedding
    # top-k overlaps the fingerprint top-k.
    assert res.within_tanimoto_mean > res.global_tanimoto_mean
    assert res.enrichment_ratio > 1.0
    assert res.spearman_emb_vs_tanimoto > 0.3
    assert res.overlap_at_k_mean > 0.3
    assert (tmp_path / f"{cfg.name}_distributions.npz").exists()


def test_nearest_molecule_by_index(tmp_path):
    store = _fingerprint_store()
    cfg = NearestMoleculeTaskConfig(k=3, query_indices=[0, 7])
    res = run_nearest_molecule(store, cfg, tmp_path)

    assert len(res.queries) == 2
    first = res.queries[0]
    assert first.query == "index:0"
    assert first.embedded
    assert len(first.neighbors) == 3
    # self is excluded and smiles are populated from the store
    assert all(n.row_index != 0 for n in first.neighbors)
    assert all(n.smiles for n in first.neighbors)
    # ranked by ascending distance
    dists = [n.distance for n in first.neighbors]
    assert dists == sorted(dists)


def test_nearest_molecule_random_sampling(tmp_path):
    store = _fingerprint_store()

    class _FakeEmbedder:
        # Return valid (non-zero) vectors so the cosine index is well-defined;
        # we only need the sampling + embed + query plumbing to run.
        def embed_atoms(self, atoms):
            return store.X[: len(atoms)].copy()

    store.embedder = _FakeEmbedder()
    cfg = NearestMoleculeTaskConfig(k=3, n_random_query_smiles=4, query_sample_seed=1)
    res = run_nearest_molecule(store, cfg, tmp_path)

    smiles_queries = [q for q in res.queries if q.query.startswith("smiles:")]
    assert len(smiles_queries) == 4
    assert all(q.embedded and len(q.neighbors) == 3 for q in smiles_queries)
    # sampling is reproducible under a fixed seed
    again = run_nearest_molecule(store, cfg, tmp_path)
    assert [q.query for q in again.queries] == [q.query for q in res.queries]


def test_nearest_molecule_requires_queries(tmp_path):
    store = _fingerprint_store()
    with pytest.raises(ValueError, match="no queries"):
        run_nearest_molecule(store, NearestMoleculeTaskConfig(k=3), tmp_path)


def test_smiles_to_atoms_generates_conformer():
    atoms = smiles_to_atoms("CCO")
    # ethanol with explicit H = 9 atoms, all with finite 3D coordinates
    assert len(atoms) == 9
    assert np.isfinite(atoms.get_positions()).all()
