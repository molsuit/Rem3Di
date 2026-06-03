"""Task A: embedding-space proximity vs chemical (Tanimoto) similarity.

Validates the embedding the way cheminformatics validates a descriptor's
"neighborhood behavior" (Patterson et al., J. Med. Chem. 1996): if the
representation is meaningful, molecules that are close *in embedding space*
should be more chemically similar than random pairs. We quantify that with:

* within-kNN Tanimoto distribution vs a global random-pair baseline,
* the enrichment (difference and ratio) between the two,
* the Spearman rank-correlation between embedding similarity and Tanimoto over
  the within-neighborhood pairs,
* the neighborhood overlap@k (Jaccard of the embedding top-k vs the
  fingerprint top-k) averaged over sampled queries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy.stats import spearmanr

from threedscriptors.evaluation.retrieval.config import TanimotoSimilarityTaskConfig
from threedscriptors.evaluation.retrieval.fingerprints import (
    bulk_tanimoto,
    morgan_fingerprints,
    tanimoto,
)
from threedscriptors.evaluation.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TanimotoSimilarityResult(BaseModel):
    task_kind: str = "tanimoto_similarity"
    name: str
    k: int
    n_queries: int
    n_within_pairs: int
    n_global_pairs: int
    within_tanimoto_mean: float
    within_tanimoto_median: float
    global_tanimoto_mean: float
    global_tanimoto_median: float
    enrichment_diff: float
    enrichment_ratio: float
    spearman_emb_vs_tanimoto: float
    spearman_pvalue: float
    overlap_at_k_mean: float


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between matched rows of ``a`` and ``b``."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=1)


def run_tanimoto_similarity(
    store: VectorStore,
    cfg: TanimotoSimilarityTaskConfig,
    output_dir: Path,
) -> TanimotoSimilarityResult:
    rng = np.random.default_rng(cfg.seed)

    fps = morgan_fingerprints(
        store.smiles, radius=cfg.fp_radius, n_bits=cfg.fp_n_bits
    )
    # Keep rows with a valid fingerprint and a parallel, non-None fp list so the
    # chemistry below never has to re-check for None (row -> valid-index map).
    valid_rows = np.array([i for i, fp in enumerate(fps) if fp is not None])
    valid_fps: list[ExplicitBitVect] = [fp for fp in fps if fp is not None]
    row_to_valid = {int(r): j for j, r in enumerate(valid_rows)}
    n_invalid = store.n - len(valid_rows)
    if n_invalid:
        logger.warning(
            "%s: %d/%d structures have unparseable SMILES and are excluded "
            "from the Tanimoto comparison.",
            cfg.name,
            n_invalid,
            store.n,
        )
    if len(valid_rows) < 2:
        raise ValueError(
            f"{cfg.name}: need >=2 structures with valid fingerprints, got "
            f"{len(valid_rows)}."
        )

    # --- sample queries -----------------------------------------------------
    n_query = min(cfg.n_query_sample, len(valid_rows))
    query_rows = rng.choice(valid_rows, size=n_query, replace=False)

    # --- within-kNN distribution + overlap@k --------------------------------
    _, nbr_idx = store.neighbors(query_rows, cfg.k)
    within_tanimoto: list[float] = []
    within_emb_sim: list[float] = []
    overlaps: list[float] = []
    for r, q_row in enumerate(query_rows):
        q_fp = valid_fps[row_to_valid[int(q_row)]]
        emb_nbrs = [int(j) for j in nbr_idx[r] if j >= 0 and int(j) in row_to_valid]
        if not emb_nbrs:
            continue
        # within-neighborhood Tanimoto + embedding cosine for each (q, nbr) pair
        q_vec = store.X[q_row][None, :]
        nbr_vecs = store.X[emb_nbrs]
        emb_sims = _cosine_rows(np.repeat(q_vec, len(emb_nbrs), axis=0), nbr_vecs)
        for nbr, esim in zip(emb_nbrs, emb_sims, strict=True):
            within_tanimoto.append(tanimoto(q_fp, valid_fps[row_to_valid[nbr]]))
            within_emb_sim.append(float(esim))

        # neighborhood overlap@k: embedding top-k vs fingerprint top-k
        sims = bulk_tanimoto(q_fp, valid_fps)
        sims[row_to_valid[int(q_row)]] = -1.0  # drop self
        fp_topk = valid_rows[np.argsort(-sims)[: cfg.k]]
        emb_set = set(emb_nbrs)
        fp_set = {int(x) for x in fp_topk}
        union = emb_set | fp_set
        overlaps.append(len(emb_set & fp_set) / len(union) if union else 0.0)

    within_arr = np.asarray(within_tanimoto, dtype=np.float64)
    emb_sim_arr = np.asarray(within_emb_sim, dtype=np.float64)

    # --- global random-pair baseline ----------------------------------------
    i_arr = rng.integers(0, len(valid_fps), size=cfg.n_global_pairs)
    j_arr = rng.integers(0, len(valid_fps), size=cfg.n_global_pairs)
    global_tanimoto = np.array(
        [
            tanimoto(valid_fps[int(i)], valid_fps[int(j)])
            for i, j in zip(i_arr, j_arr, strict=True)
            if i != j
        ],
        dtype=np.float64,
    )

    # --- aggregate ----------------------------------------------------------
    within_mean = float(within_arr.mean())
    global_mean = float(global_tanimoto.mean())
    if emb_sim_arr.size >= 2 and np.ptp(within_arr) > 0 and np.ptp(emb_sim_arr) > 0:
        rho, pval = spearmanr(emb_sim_arr, within_arr)
    else:
        rho, pval = float("nan"), float("nan")

    np.savez_compressed(
        Path(output_dir) / f"{cfg.name}_distributions.npz",
        within_tanimoto=within_arr,
        within_emb_sim=emb_sim_arr,
        global_tanimoto=global_tanimoto,
    )

    result = TanimotoSimilarityResult(
        name=cfg.name,
        k=cfg.k,
        n_queries=n_query,
        n_within_pairs=int(within_arr.size),
        n_global_pairs=int(global_tanimoto.size),
        within_tanimoto_mean=within_mean,
        within_tanimoto_median=float(np.median(within_arr)),
        global_tanimoto_mean=global_mean,
        global_tanimoto_median=float(np.median(global_tanimoto)),
        enrichment_diff=within_mean - global_mean,
        enrichment_ratio=within_mean / global_mean if global_mean > 0 else float("inf"),
        spearman_emb_vs_tanimoto=float(rho),
        spearman_pvalue=float(pval),
        overlap_at_k_mean=float(np.mean(overlaps)) if overlaps else 0.0,
    )
    logger.info(
        "%s: within-kNN Tanimoto=%.3f vs global=%.3f (ratio %.2fx), "
        "Spearman(emb,Tanimoto)=%.3f, overlap@%d=%.3f",
        cfg.name,
        result.within_tanimoto_mean,
        result.global_tanimoto_mean,
        result.enrichment_ratio,
        result.spearman_emb_vs_tanimoto,
        cfg.k,
        result.overlap_at_k_mean,
    )
    return result
