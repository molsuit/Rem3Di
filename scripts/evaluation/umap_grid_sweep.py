"""Sweep UMAP hyperparameters on cached descriptors and save scatter plots.

Loads ``descriptors.pt`` from a model's ``analysis/`` directory, L2-normalizes (to
match :class:`DescriptorAnalysisRunner`'s default), computes metal-center
colors, then fits a grid of UMAP configurations and saves one PNG per
combination plus the raw 2D coordinates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis.tmqm_clustering_utils import (
    get_atomic_num_colors,
    get_block_colors,
    get_metal_center_type,
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _scatter(
    coords: np.ndarray,
    colors,
    handles: list[Line2D] | None,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=2,
        alpha=0.6,
        rasterized=True,
        linewidths=0,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    if handles is not None:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_grid(arg: str, cast) -> list:
    return [cast(x) for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptors-path",
        type=Path,
        required=True,
        help="Path to descriptors.pt (typically <model_dir>/analysis/descriptors.pt).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/scratch/s5f/wedigs.s5f/datasets/tmqm"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the grid of PNGs and projections.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=str,
        default="5,10,15,30,50,100",
        help="Comma-separated list.",
    )
    parser.add_argument(
        "--min-dist", type=str, default="0.0,0.1,0.3", help="Comma-separated list."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean,cosine",
        help="Comma-separated list (any umap-learn metric).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Pinned for reproducibility (UMAP runs single-threaded when set).",
    )
    parser.add_argument(
        "--save-coords",
        action="store_true",
        help="Also save the 2D projection as .npy alongside each PNG.",
    )
    args = parser.parse_args()

    n_neighbors_list = parse_grid(args.n_neighbors, int)
    min_dist_list = parse_grid(args.min_dist, float)
    metric_list = parse_grid(args.metric, str)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading descriptors from {args.descriptors_path}")
    raw = torch.load(args.descriptors_path, map_location="cpu")
    if isinstance(raw, torch.Tensor):
        descriptors = raw.detach().cpu().numpy()
    else:
        descriptors = np.asarray(raw)
    descriptors = descriptors.astype(np.float64, copy=False)
    print(f"Descriptors shape: {descriptors.shape}")
    descriptors = _l2_normalize(descriptors)

    print(f"Loading dataset from {args.dataset_dir}")
    dataset = MoleculeDataset.open_existing_dataset_from_dir(args.dataset_dir)
    molecules = dataset.get_all_molecules()
    if len(molecules) != descriptors.shape[0]:
        raise ValueError(
            f"Dataset has {len(molecules)} molecules but descriptors has "
            f"{descriptors.shape[0]} rows."
        )

    print("Computing metal-center colors")
    atomic_nums = get_metal_center_type(molecules)
    element_colors, element_handles = get_atomic_num_colors(atomic_nums)
    block_colors = get_block_colors(atomic_nums)

    grid = [
        (nn, md, m)
        for m in metric_list
        for nn in n_neighbors_list
        for md in min_dist_list
    ]
    print(f"Sweeping {len(grid)} UMAP configurations")

    log = []
    for idx, (nn, md, metric) in enumerate(grid, start=1):
        tag = f"{metric}_nn{nn}_md{md:g}"
        print(f"[{idx}/{len(grid)}] {tag}")
        t0 = time.perf_counter()
        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=nn,
                min_dist=md,
                metric=metric,
                random_state=args.random_state,
            )
            coords = reducer.fit_transform(descriptors)
            coords = coords - coords.mean(axis=0, keepdims=True)
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(f"  FAILED in {elapsed:.1f}s: {exc}")
            log.append(
                {
                    "tag": tag,
                    "n_neighbors": nn,
                    "min_dist": md,
                    "metric": metric,
                    "elapsed_s": elapsed,
                    "error": repr(exc),
                }
            )
            continue

        title = f"UMAP n_neighbors={nn} min_dist={md} metric={metric}"
        _scatter(
            coords,
            element_colors,
            element_handles,
            f"{title} (metal element)",
            args.output_dir / f"{tag}_element.png",
        )
        _scatter(
            coords,
            block_colors,
            None,
            f"{title} (d-block)",
            args.output_dir / f"{tag}_block.png",
        )
        if args.save_coords:
            np.save(args.output_dir / f"{tag}.npy", coords.astype(np.float32))

        log.append(
            {
                "tag": tag,
                "n_neighbors": nn,
                "min_dist": md,
                "metric": metric,
                "elapsed_s": elapsed,
                "x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
                "y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
            }
        )
        print(f"  done in {elapsed:.1f}s")

    with (args.output_dir / "grid_log.json").open("w") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote log to {args.output_dir / 'grid_log.json'}")


if __name__ == "__main__":
    main()
