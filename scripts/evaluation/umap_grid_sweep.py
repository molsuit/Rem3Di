"""Sweep UMAP hyperparameters on cached descriptors and save scatter plots.

Loads ``descriptors.pt`` from a model's ``analysis/`` directory, L2-normalizes
(to match :class:`DescriptorAnalysisRunner`'s default), then fits a grid of
UMAP configurations. Each projection is rasterized with datashader using
``count_cat`` aggregation and ``how='eq_hist'`` shading to mitigate
overplotting saturation in dense regions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import umap
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from datashader.utils import export_image

from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.descriptor_analysis.tmqm_clustering_utils import (
    get_metal_center_type,
)

BLOCK_DEFINITIONS: dict[str, list[str]] = {
    "3d": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    "4d": ["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"],
    "5d": ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
    "f": ["La"],
}
BLOCK_COLOR_KEY: dict[str, str] = {
    "3d": "#1f77b4",
    "4d": "#2ca02c",
    "5d": "#d62728",
    "f": "#ff7f0e",
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _build_categories(
    atomic_nums: list[int],
) -> tuple[pd.Categorical, dict[str, str], pd.Categorical, dict[str, str]]:
    symbols = [chemical_symbols[n] for n in atomic_nums]
    unique_symbols = sorted(set(symbols), key=lambda s: chemical_symbols.index(s))
    elem_color_key = {
        s: mcolors.to_hex(jmol_colors[chemical_symbols.index(s)])
        for s in unique_symbols
    }
    elem_categorical = pd.Categorical(symbols, categories=unique_symbols)

    symbol_to_block = {
        sym: blk for blk, syms in BLOCK_DEFINITIONS.items() for sym in syms
    }
    blocks = [symbol_to_block[s] for s in symbols]
    present_blocks = [b for b in BLOCK_DEFINITIONS if b in set(blocks)]
    block_color_key = {b: BLOCK_COLOR_KEY[b] for b in present_blocks}
    block_categorical = pd.Categorical(blocks, categories=present_blocks)

    return elem_categorical, elem_color_key, block_categorical, block_color_key


def _datashade_categorical(
    coords: np.ndarray,
    category: pd.Categorical,
    color_key: dict[str, str],
    *,
    plot_size: int = 1400,
    how: str = "eq_hist",
    min_alpha: int = 120,
    spread_threshold: float = 0.5,
    spread_max_px: int = 4,
) -> tf.Image:
    df = pd.DataFrame(
        {
            "x": coords[:, 0].astype(np.float32),
            "y": coords[:, 1].astype(np.float32),
            "cat": category,
        }
    )
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())
    pad_x = 0.03 * (x_max - x_min + 1e-9)
    pad_y = 0.03 * (y_max - y_min + 1e-9)
    canvas = ds.Canvas(
        plot_width=plot_size,
        plot_height=plot_size,
        x_range=(x_min - pad_x, x_max + pad_x),
        y_range=(y_min - pad_y, y_max + pad_y),
    )
    agg = canvas.points(df, "x", "y", ds.count_cat("cat"))
    img = tf.shade(agg, color_key=color_key, how=how, min_alpha=min_alpha)
    if spread_max_px > 0:
        img = tf.dynspread(img, threshold=spread_threshold, max_px=spread_max_px)
    return tf.set_background(img, "white")


def _save_image(img: tf.Image, out_path: Path) -> None:
    export_image(
        img,
        filename=out_path.stem,
        fmt=".png",
        background="white",
        export_path=str(out_path.parent),
    )


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
        default=Path("/path/to/datasets/tmqm"),
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
    parser.add_argument(
        "--reuse-cached-coords",
        action="store_true",
        help="If <output-dir>/<tag>.npy exists, load it and skip the UMAP fit. "
        "Useful for re-rendering plots when only the plotting code changed.",
    )
    parser.add_argument(
        "--plot-size",
        type=int,
        default=1400,
        help="Datashader canvas size in pixels (square). Output figsize and DPI "
        "are matched so the saved axes region maps ~1:1 to the canvas.",
    )
    parser.add_argument(
        "--shade-how",
        type=str,
        default="eq_hist",
        choices=["eq_hist", "log", "linear", "cbrt"],
        help="Datashader density transfer function.",
    )
    parser.add_argument(
        "--min-alpha",
        type=int,
        default=120,
        help="Minimum per-point alpha (0-255). Higher = more solid sparse points.",
    )
    parser.add_argument(
        "--spread-threshold",
        type=float,
        default=0.5,
        help="tf.dynspread threshold: minimum fraction of adjacent non-empty pixels "
        "before spreading stops growing.",
    )
    parser.add_argument(
        "--spread-max-px",
        type=int,
        default=4,
        help="tf.dynspread max pixel radius. 0 disables spreading.",
    )
    args = parser.parse_args()

    n_neighbors_list = parse_grid(args.n_neighbors, int)
    min_dist_list = parse_grid(args.min_dist, float)
    metric_list = parse_grid(args.metric, str)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid = [
        (nn, md, m)
        for m in metric_list
        for nn in n_neighbors_list
        for md in min_dist_list
    ]
    all_cached = args.reuse_cached_coords and all(
        (args.output_dir / f"{metric}_nn{nn}_md{md:g}.npy").exists()
        for (nn, md, metric) in grid
    )

    atomic_nums_cache = args.output_dir / "metal_atomic_nums.npy"
    descriptors: np.ndarray | None = None
    atomic_nums: list[int]

    if all_cached and atomic_nums_cache.exists():
        print(
            f"All UMAP configs cached and {atomic_nums_cache.name} present; "
            "skipping descriptors and dataset loading"
        )
        atomic_nums = np.load(atomic_nums_cache).tolist()
    else:
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
        atomic_nums = get_metal_center_type(molecules)
        np.save(atomic_nums_cache, np.asarray(atomic_nums, dtype=np.int32))

    print("Building categorical color maps")
    elem_cat, elem_key, block_cat, block_key = _build_categories(atomic_nums)
    expected_rows = len(atomic_nums)

    print(f"Sweeping {len(grid)} UMAP configurations")

    log = []
    for idx, (nn, md, metric) in enumerate(grid, start=1):
        tag = f"{metric}_nn{nn}_md{md:g}"
        print(f"[{idx}/{len(grid)}] {tag}")
        cached_path = args.output_dir / f"{tag}.npy"
        cached = args.reuse_cached_coords and cached_path.exists()
        t0 = time.perf_counter()
        if cached:
            coords = np.load(cached_path).astype(np.float64)
            if coords.shape != (expected_rows, 2):
                raise ValueError(
                    f"Cached coords {cached_path} shape {coords.shape} does not "
                    f"match expected row count {expected_rows}"
                )
            elapsed = time.perf_counter() - t0
            print(f"  loaded cached coords in {elapsed:.2f}s")
        else:
            if descriptors is None:
                raise RuntimeError(
                    "Descriptors were not loaded but a UMAP fit is required for "
                    f"{tag}. This indicates a stale cache state."
                )
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
            except Exception as exc:
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

        elem_img = _datashade_categorical(
            coords,
            elem_cat,
            elem_key,
            plot_size=args.plot_size,
            how=args.shade_how,
            spread_threshold=args.spread_threshold,
            spread_max_px=args.spread_max_px,
            min_alpha=args.min_alpha,
        )
        _save_image(elem_img, args.output_dir / f"{tag}_element.png")

        block_img = _datashade_categorical(
            coords,
            block_cat,
            block_key,
            plot_size=args.plot_size,
            how=args.shade_how,
            spread_threshold=args.spread_threshold,
            spread_max_px=args.spread_max_px,
            min_alpha=args.min_alpha,
        )
        _save_image(block_img, args.output_dir / f"{tag}_block.png")

        if args.save_coords and not cached:
            np.save(cached_path, coords.astype(np.float32))

        log.append(
            {
                "tag": tag,
                "n_neighbors": nn,
                "min_dist": md,
                "metric": metric,
                "elapsed_s": elapsed,
                "cached": cached,
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
