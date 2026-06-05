"""UMAP-project the cached REM3DI embeddings and color by a QM9 property.

Reads the embedding matrices cached by ``run_eval.py``
(``{cache_dir}/{dataset_id}__{model}.npz``) -- so this runs on a login node with
no GPU and no re-embedding -- projects each to 2D with UMAP (cosine metric, to
match the retrieval index), and renders one scatter panel per model on a shared
color scale. The default coloring is the HOMO-LUMO gap: the natural electronic
coordinate for a 3D/electronic descriptor, so the panel shows whether the
embedding organizes molecules by electronic structure rather than only by 2D
scaffold. ``--color-by`` accepts any QM9 task name (``gap``, ``alpha``, ``mu``,
...) or ``size`` (heavy-atom count).

The 2D coordinates + colors are also written to an ``.npz`` so the figure can be
re-styled without re-fitting UMAP.

Run with:
    uv run python scripts/evaluation/embedding_umap.py
    uv run python scripts/evaluation/embedding_umap.py --color-by alpha
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("remedi_pcqm_agg_mean", "pcqm_agg_mean"),
    ("remedi_pcqm_baseline", "pcqm_baseline"),
]


def color_values(dataset: MoleculeDataset, color_by: str) -> tuple[np.ndarray, str]:
    """Per-structure scalar to color by, plus a colorbar label."""
    n = dataset.N_structures
    if color_by == "size":
        ptr = np.asarray(dataset.ptr[: n + 1], dtype=np.int64)
        return np.diff(ptr).astype(np.float64), "heavy+H atom count"
    smap = dataset.config.tasks.system_map
    if color_by not in smap:
        raise ValueError(
            f"--color-by {color_by!r} not a QM9 task; choose 'size' or one of "
            f"{sorted(smap)}."
        )
    vals = np.asarray(dataset.targets_system[:n], dtype=np.float64)[:, smap[color_by]]
    return vals, f"{color_by}"


def fit_umap(
    X: np.ndarray, *, n_neighbors: int, min_dist: float, metric: str, seed: int
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_components=2,
    )
    return reducer.fit_transform(np.ascontiguousarray(X, dtype=np.float32))


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
            "/p/scratch/mace/wedig1/evaluation_results/qm9_retrieval/embedding_umap.png"
        ),
    )
    p.add_argument("--color-by", type=str, default="gap")
    p.add_argument(
        "--n-sample",
        type=int,
        default=50_000,
        help="Subsample size fed to UMAP (<=0 uses all structures).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-neighbors", type=int, default=50)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MoleculeDataset.open_existing_dataset_from_dir(args.dataset_path)
    n = dataset.N_structures
    colors_full, clabel = color_values(dataset, args.color_by)

    rng = np.random.default_rng(args.seed)
    if args.n_sample and 0 < args.n_sample < n:
        sample_idx = np.sort(rng.choice(n, size=args.n_sample, replace=False))
    else:
        sample_idx = np.arange(n)
    colors = colors_full[sample_idx]
    # Robust shared color scale across both panels (clip 1st/99th percentile).
    vmin, vmax = np.percentile(colors, [1, 99])

    coords_by_model: dict[str, np.ndarray] = {}
    for cache_name, label in DEFAULT_MODELS:
        cache_path = args.cache_dir / f"{args.dataset_id}__{cache_name}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached embeddings: {cache_path}. Run the eval first."
            )
        X = np.asarray(np.load(cache_path)["X"])
        if X.shape[0] != n:
            raise ValueError(
                f"{label}: embedding rows ({X.shape[0]}) != structures ({n})."
            )
        print(f"fitting UMAP for {label} ({len(sample_idx)} points, {X.shape[1]}d)...")
        coords_by_model[label] = fit_umap(
            X[sample_idx],
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            seed=args.seed,
        )

    fig, axes = plt.subplots(
        1, len(coords_by_model), figsize=(7.0 * len(coords_by_model), 6.0)
    )
    axes = np.atleast_1d(axes)
    scatter = None
    for ax, (label, coords) in zip(axes, coords_by_model.items(), strict=True):
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=3,
            alpha=0.5,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(f"{label} REM3DI embedding")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(scatter, ax=list(axes), label=clabel, shrink=0.8)
    fig.suptitle(
        f"REM3DI embedding UMAP (metric={args.metric}, n={len(sample_idx)}), "
        f"colored by {clabel}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        sample_idx=sample_idx,
        colors=colors,
        **{f"coords_{label}": c for label, c in coords_by_model.items()},
    )
    print(f"wrote {args.output}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
