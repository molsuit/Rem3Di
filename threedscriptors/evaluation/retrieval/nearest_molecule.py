"""Task C: nearest-molecule retrieval.

Given a query — an existing store entry (by row index) or a brand-new molecule
(by SMILES) — return the closest dataset molecules. New SMILES are embedded
through the *same* trained model that built the store, using a generated 3D
conformer.

Caveat on new-SMILES queries: REM3DI is geometry-sensitive and the dataset was
embedded from its own (here DFT-derived) conformers. A query conformer is
generated with RDKit ETKDG + MMFF, which differs from a DFT geometry, so a
new-SMILES query carries some geometry-induced distribution shift. It is fine
for "what's the most similar known molecule to A" style retrieval; for the most
faithful comparison, query with the same quality of geometry the store was
built from.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem

from threedscriptors.evaluation.retrieval.config import NearestMoleculeTaskConfig
from threedscriptors.evaluation.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    rank: int
    row_index: int
    structure_id: int
    smiles: str
    distance: float


class QueryResult(BaseModel):
    query: str  # "smiles:<...>" or "index:<i>"
    embedded: bool  # False if a new-SMILES query could not be embedded
    neighbors: list[Neighbor]


class NearestMoleculeResult(BaseModel):
    task_kind: str = "nearest_molecule"
    name: str
    k: int
    queries: list[QueryResult]


def smiles_to_atoms(smiles: str, *, seed: int = 0xF00D) -> Atoms:
    """SMILES -> RDKit ETKDG conformer (+ MMFF) -> ASE Atoms (explicit H)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"ETKDG conformer generation failed for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    return Atoms(
        numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
        positions=conf.GetPositions(),
    )


def _neighbors_for(
    store: VectorStore, dist_row: np.ndarray, idx_row: np.ndarray
) -> list[Neighbor]:
    return [
        Neighbor(
            rank=rank,
            row_index=int(row),
            structure_id=int(store.structure_ids[int(row)]),
            smiles=store.smiles[int(row)],
            distance=float(d),
        )
        for rank, (row, d) in enumerate(zip(idx_row, dist_row, strict=True))
        if row >= 0
    ]


def _index_queries(
    store: VectorStore, cfg: NearestMoleculeTaskConfig
) -> list[QueryResult]:
    """Neighbors of existing store entries (by row index), excluding self."""
    if not cfg.query_indices:
        return []
    idx = np.asarray(cfg.query_indices, dtype=np.int64)
    if (idx < 0).any() or (idx >= store.n).any():
        raise ValueError(f"{cfg.name}: query_indices out of range [0, {store.n}).")
    dist, nbr = store.neighbors(idx, cfg.k)  # exclude self
    return [
        QueryResult(
            query=f"index:{int(q)}",
            embedded=True,
            neighbors=_neighbors_for(store, dist[r], nbr[r]),
        )
        for r, q in enumerate(idx)
    ]


def _resolve_query_smiles(
    store: VectorStore, cfg: NearestMoleculeTaskConfig
) -> list[str]:
    """Explicit query SMILES plus N sampled uniformly at random from the store."""
    query_smiles = list(cfg.query_smiles)
    if cfg.n_random_query_smiles <= 0:
        return query_smiles
    valid = [s for s in store.smiles if s]
    rng = np.random.default_rng(cfg.query_sample_seed)
    n_sample = min(cfg.n_random_query_smiles, len(valid))
    if n_sample < cfg.n_random_query_smiles:
        logger.warning(
            "%s: requested %d random query SMILES but only %d are available.",
            cfg.name,
            cfg.n_random_query_smiles,
            n_sample,
        )
    picked = rng.choice(len(valid), size=n_sample, replace=False)
    query_smiles.extend(valid[int(i)] for i in picked)
    return query_smiles


def _smiles_queries(
    store: VectorStore, cfg: NearestMoleculeTaskConfig
) -> list[QueryResult]:
    """Embed each query SMILES (fresh conformer) and retrieve nearest neighbors."""
    query_smiles = _resolve_query_smiles(store, cfg)
    if not query_smiles:
        return []

    results: list[QueryResult] = []
    embeddable: list[tuple[str, Atoms]] = []
    for smi in query_smiles:
        try:
            embeddable.append((smi, smiles_to_atoms(smi, seed=cfg.conformer_seed)))
        except (ValueError, RuntimeError) as exc:
            logger.warning("%s: skipping query %r (%s)", cfg.name, smi, exc)
            results.append(
                QueryResult(query=f"smiles:{smi}", embedded=False, neighbors=[])
            )
    if embeddable:
        vecs = store.embed_atoms([a for _, a in embeddable])
        dist, nbr = store.query(vecs, cfg.k)
        for r, (smi, _) in enumerate(embeddable):
            results.append(
                QueryResult(
                    query=f"smiles:{smi}",
                    embedded=True,
                    neighbors=_neighbors_for(store, dist[r], nbr[r]),
                )
            )
    return results


def run_nearest_molecule(
    store: VectorStore,
    cfg: NearestMoleculeTaskConfig,
    output_dir: Path,
) -> NearestMoleculeResult:
    del output_dir  # results are serialized by the runner
    if not cfg.has_queries():
        raise ValueError(
            f"{cfg.name}: no queries given (set query_smiles, query_indices, or "
            f"n_random_query_smiles)."
        )

    queries = _index_queries(store, cfg) + _smiles_queries(store, cfg)
    logger.info(
        "%s: retrieved top-%d neighbors for %d queries", cfg.name, cfg.k, len(queries)
    )
    return NearestMoleculeResult(name=cfg.name, k=cfg.k, queries=queries)
