"""Plot how within-kNN Tanimoto similarity decays as k grows, with floor + ceiling.

The retrieval Tanimoto task (``tanimoto_similarity``) reports the within-kNN
Tanimoto at a single ``k`` (10 in the QM9 manifests). This script sweeps ``k``
and plots the curve: for each query, take its top-k embedding-space neighbors and
average their Morgan/Tanimoto similarity to the query, then average over queries.

To make the number interpretable it brackets the embedding curve with two
reference curves:

* **floor** -- the global random-pair Tanimoto mean (k-independent). What you'd
  get from a representation with no chemical structure.
* **ceiling** -- retrieve each query's top-k neighbors *by Tanimoto itself*
  (fingerprint-space ranking) and average their Tanimoto. This is the maximum
  attainable within-kNN Tanimoto at each k, so the embedding curve necessarily
  lies between floor and ceiling. The gap to the ceiling is the chemical-
  similarity signal a pure-fingerprint retriever captures that the embedding
  does not. The ceiling depends only on fingerprints + sampled queries, so it is
  identical for every model.

It reuses the *cached* embedding matrices written by ``run_eval.py``
(``{cache_dir}/{dataset_id}__{model}.npz``), so it runs on a login node with no
GPU and no re-embedding. SMILES + neighbor sampling mirror the eval task
(``np.random.default_rng(seed)``, self-match dropped).

Run with:
    uv run python scripts/evaluation/tanimoto_k_sweep.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.retrieval.fingerprints import (
    bulk_tanimoto,
    morgan_fingerprints,
    tanimoto,
)
from remedi.evaluation.retrieval.vector_store import (
    SklearnFlatIndex,
    VectorStore,
)

# (cache-name, display label) for the models to overlay on one plot.
DEFAULT_MODELS: list[tuple[str, str]] = [
    ("remedi_pcqm_agg_mean", "pcqm_agg_mean"),
    ("remedi_pcqm_baseline", "pcqm_baseline"),
]
DEFAULT_KS: list[int] = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]


@dataclass
class Curve:
    """A within-kNN Tanimoto curve over k (mean + per-query SEM)."""

    label: str
    ks: list[int]
    mean: list[float]
    sem: list[float]


def _valid_fingerprints(
    smiles: list[str], radius: int, n_bits: int
) -> tuple[np.ndarray, list[ExplicitBitVect], dict[int, int]]:
    fps = morgan_fingerprints(smiles, radius=radius, n_bits=n_bits)
    valid_rows = np.array([i for i, fp in enumerate(fps) if fp is not None])
    valid_fps = [fp for fp in fps if fp is not None]
    row_to_valid = {int(r): j for j, r in enumerate(valid_rows)}
    return valid_rows, valid_fps, row_to_valid


def _cumulative_curve(label: str, per_query_tani: np.ndarray, ks: list[int]) -> Curve:
    """Per-query Tanimoto-by-rank matrix -> mean + SEM at each k.

    Queries are the independent units: the statistic at k is each query's mean
    over its first-k neighbors, then mean/SEM across queries.
    """
    mean, sem = [], []
    for k in ks:
        per_query_k = np.nanmean(per_query_tani[:, :k], axis=1)
        n = int(np.sum(~np.isnan(per_query_k)))
        mean.append(float(np.nanmean(per_query_k)))
        sem.append(float(np.nanstd(per_query_k) / np.sqrt(n)) if n else float("nan"))
    return Curve(label=label, ks=list(ks), mean=mean, sem=sem)


def embedding_curve(
    X: np.ndarray,
    smiles: list[str],
    label: str,
    *,
    query_rows: np.ndarray,
    valid_fps: list[ExplicitBitVect],
    row_to_valid: dict[int, int],
    ks: list[int],
    metric: str,
) -> Curve:
    """within-kNN Tanimoto when neighbors are ranked by embedding distance."""
    store = VectorStore(
        X=np.ascontiguousarray(X, dtype=np.float32),
        smiles=smiles,
        structure_ids=np.arange(len(smiles)),
        index=SklearnFlatIndex(metric=metric),
        embedder=None,
    )
    store.index.fit(store.X)

    k_max = max(ks)
    _, nbr_idx = store.neighbors(query_rows, k_max)
    per_query_tani = np.full((len(query_rows), k_max), np.nan, dtype=np.float64)
    for r, q_row in enumerate(query_rows):
        q_fp = valid_fps[row_to_valid[int(q_row)]]
        for rank, j in enumerate(nbr_idx[r]):
            j = int(j)
            if j >= 0 and j in row_to_valid:
                per_query_tani[r, rank] = tanimoto(q_fp, valid_fps[row_to_valid[j]])
    return _cumulative_curve(label, per_query_tani, ks)


def ceiling_curve(
    *,
    query_rows: np.ndarray,
    valid_fps: list[ExplicitBitVect],
    row_to_valid: dict[int, int],
    ks: list[int],
) -> Curve:
    """Max-attainable within-kNN Tanimoto: neighbors ranked by Tanimoto itself."""
    k_max = max(ks)
    per_query_tani = np.full((len(query_rows), k_max), np.nan, dtype=np.float64)
    for r, q_row in enumerate(query_rows):
        j_self = row_to_valid[int(q_row)]
        q_fp = valid_fps[j_self]
        sims = bulk_tanimoto(q_fp, valid_fps)
        sims[j_self] = -1.0  # drop self before taking the top-k
        top = np.argpartition(-sims, k_max)[:k_max]
        top = top[np.argsort(-sims[top])]  # sort the top slice descending
        per_query_tani[r, :] = sims[top]
    return _cumulative_curve("Tanimoto top-k (ceiling)", per_query_tani, ks)


def global_baseline(
    valid_fps: list[ExplicitBitVect], n_global_pairs: int, rng: np.random.Generator
) -> float:
    """Mean Tanimoto over random molecule pairs -- the floor."""
    i_arr = rng.integers(0, len(valid_fps), size=n_global_pairs)
    j_arr = rng.integers(0, len(valid_fps), size=n_global_pairs)
    vals = [
        tanimoto(valid_fps[int(i)], valid_fps[int(j)])
        for i, j in zip(i_arr, j_arr, strict=True)
        if i != j
    ]
    return float(np.mean(vals))


def plot_sweeps(
    models: list[Curve],
    ceiling: Curve,
    floor: float,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Ceiling (model-independent) as a shaded upper envelope.
    ax.errorbar(
        ceiling.ks,
        ceiling.mean,
        yerr=ceiling.sem,
        marker="s",
        capsize=3,
        color="black",
        label="Tanimoto top-k (ceiling)",
    )
    for i, res in enumerate(models):
        ax.errorbar(
            res.ks,
            res.mean,
            yerr=res.sem,
            marker="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{res.label} (embedding kNN)",
        )
    ax.axhline(
        floor, ls="--", lw=1.2, color="grey", label="random-pair baseline (floor)"
    )

    ax.set_xscale("log")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("mean within-kNN Tanimoto similarity")
    ax.set_title("Chemical (Tanimoto) similarity vs neighborhood size k")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/p/scratch/mace/wedig1/evaluation_results/qm9_retrieval/descriptor_cache"
        ),
    )
    p.add_argument(
        "--dataset-path", type=Path, default=Path("/p/scratch/mace/wedig1/datasets/qm9")
    )
    p.add_argument("--dataset-id", type=str, default="qm9")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/p/scratch/mace/wedig1/evaluation_results/qm9_retrieval/tanimoto_k_sweep.png"
        ),
    )
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--n-query", type=int, default=500)
    p.add_argument("--n-global-pairs", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-n-bits", type=int, default=2048)
    p.add_argument("--metric", type=str, default="cosine")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MoleculeDataset.open_existing_dataset_from_dir(args.dataset_path)
    smiles = dataset.get_smiles_per_structure()

    # Shared across models: fingerprints, sampled queries, floor, ceiling.
    valid_rows, valid_fps, row_to_valid = _valid_fingerprints(
        smiles, args.fp_radius, args.fp_n_bits
    )
    if len(valid_rows) < 2:
        raise ValueError("<2 valid fingerprints.")
    rng = np.random.default_rng(args.seed)
    n_query = min(args.n_query, len(valid_rows))
    query_rows = rng.choice(valid_rows, size=n_query, replace=False)
    floor = global_baseline(valid_fps, args.n_global_pairs, rng)
    ceiling = ceiling_curve(
        query_rows=query_rows,
        valid_fps=valid_fps,
        row_to_valid=row_to_valid,
        ks=args.ks,
    )

    models: list[Curve] = []
    for cache_name, label in DEFAULT_MODELS:
        cache_path = args.cache_dir / f"{args.dataset_id}__{cache_name}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached embeddings: {cache_path}. Run the eval first."
            )
        X = np.asarray(np.load(cache_path)["X"])
        if X.shape[0] != len(smiles):
            raise ValueError(
                f"{label}: embedding rows ({X.shape[0]}) != dataset structures "
                f"({len(smiles)})."
            )
        models.append(
            embedding_curve(
                X,
                smiles,
                label,
                query_rows=query_rows,
                valid_fps=valid_fps,
                row_to_valid=row_to_valid,
                ks=args.ks,
                metric=args.metric,
            )
        )

    plot_sweeps(models, ceiling, floor, args.output)

    rows = []
    for res in [ceiling, *models]:
        for i, k in enumerate(res.ks):
            rows.append(
                {
                    "curve": res.label,
                    "k": k,
                    "within_tanimoto_mean": res.mean[i],
                    "within_tanimoto_sem": res.sem[i],
                    "floor_random_pair_mean": floor,
                }
            )
    csv_path = args.output.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"wrote {args.output}")
    print(f"wrote {csv_path}")
    print(f"floor (random-pair Tanimoto mean): {floor:.3f}")
    for res in [ceiling, *models]:
        pairs = ", ".join(
            f"k={k}:{m:.3f}" for k, m in zip(res.ks, res.mean, strict=True)
        )
        print(f"{res.label}: {pairs}")


if __name__ == "__main__":
    main()
