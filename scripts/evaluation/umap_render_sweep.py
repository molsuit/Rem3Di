"""Sweep datashader render parameters on a fixed cached UMAP projection.

Loads a single ``<tag>.npy`` coord file plus the sibling ``metal_atomic_nums.npy``
written by ``umap_grid_sweep.py``, then renders the same projection under a grid
of ``(spread_max_px, spread_threshold, min_alpha)`` settings to make the visual
trade-offs easy to compare side-by-side. PNGs are written via
``datashader.utils.export_image`` so they are not downscaled by matplotlib.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from umap_grid_sweep import (  # type: ignore[import-not-found]
    _build_categories,
    _datashade_categorical,
    _save_image,
)

BLOCK_PALETTES: dict[str, dict[str, str]] = {
    "default": {"3d": "#1f77b4", "4d": "#2ca02c", "5d": "#d62728", "f": "#ff7f0e"},
    "bright": {"3d": "#0066FF", "4d": "#00C800", "5d": "#FF0000", "f": "#FF00FF"},
    "set1": {"3d": "#377eb8", "4d": "#4daf4a", "5d": "#e41a1c", "f": "#984ea3"},
    "okabe": {"3d": "#0072B2", "4d": "#009E73", "5d": "#D55E00", "f": "#CC79A7"},
    "cud": {"3d": "#56B4E9", "4d": "#009E73", "5d": "#E69F00", "f": "#CC79A7"},
}


def parse_grid(arg: str, cast):
    return [cast(x) for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coords-path",
        type=Path,
        required=True,
        help="Path to a cached <tag>.npy coord file from umap_grid_sweep.py.",
    )
    parser.add_argument(
        "--atomic-nums-path",
        type=Path,
        required=True,
        help="Path to metal_atomic_nums.npy written by umap_grid_sweep.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the render-sweep PNGs.",
    )
    parser.add_argument(
        "--plot-size",
        type=int,
        default=2000,
        help="Datashader canvas size in pixels (square).",
    )
    parser.add_argument(
        "--shade-how",
        type=str,
        default="eq_hist",
        choices=["eq_hist", "log", "linear", "cbrt"],
    )
    parser.add_argument(
        "--spread-max-px",
        type=str,
        default="2,4,6,8",
        help="Comma-separated list of tf.dynspread max-radius values.",
    )
    parser.add_argument(
        "--spread-threshold",
        type=str,
        default="0.5,0.75,0.95",
        help="Comma-separated list of tf.dynspread threshold values.",
    )
    parser.add_argument(
        "--min-alpha",
        type=str,
        default="120,160,200",
        help="Comma-separated list of tf.shade min_alpha values (0-255).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="element",
        choices=["element", "block", "both"],
        help="Which categorical color map to render.",
    )
    parser.add_argument(
        "--block-palette",
        type=str,
        default="default",
        help=(
            "Comma-separated list of palette names for the block category. "
            f"Available: {','.join(BLOCK_PALETTES)}. Ignored for element category."
        ),
    )
    args = parser.parse_args()

    max_px_list = parse_grid(args.spread_max_px, int)
    threshold_list = parse_grid(args.spread_threshold, float)
    min_alpha_list = parse_grid(args.min_alpha, int)
    palette_list = parse_grid(args.block_palette, str)
    for name in palette_list:
        if name not in BLOCK_PALETTES:
            raise ValueError(
                f"Unknown block palette {name!r}. Available: {list(BLOCK_PALETTES)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    coords = np.load(args.coords_path).astype(np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) coords, got shape {coords.shape}")
    atomic_nums = np.load(args.atomic_nums_path).tolist()
    if len(atomic_nums) != coords.shape[0]:
        raise ValueError(
            f"atomic_nums has {len(atomic_nums)} entries but coords has "
            f"{coords.shape[0]} rows."
        )

    elem_cat, elem_key, block_cat, _block_key_default = _build_categories(atomic_nums)
    categories: list[tuple[str, object, dict[str, str]]] = []
    if args.category in ("element", "both"):
        categories.append(("element", elem_cat, elem_key))
    if args.category in ("block", "both"):
        for palette_name in palette_list:
            block_key = {
                blk: BLOCK_PALETTES[palette_name][blk] for blk in block_cat.categories
            }
            categories.append((f"block-{palette_name}", block_cat, block_key))

    base_tag = args.coords_path.stem
    grid = list(itertools.product(max_px_list, threshold_list, min_alpha_list))
    print(
        f"Rendering {len(grid)} configs x {len(categories)} categories from "
        f"{base_tag} -> {args.output_dir}"
    )

    for idx, (max_px, threshold, min_alpha) in enumerate(grid, start=1):
        cfg_tag = f"px{max_px}_th{threshold:g}_a{min_alpha}"
        print(f"[{idx}/{len(grid)}] {cfg_tag}")
        for cat_name, cat, color_key in categories:
            img = _datashade_categorical(
                coords,
                cat,
                color_key,
                plot_size=args.plot_size,
                how=args.shade_how,
                spread_threshold=threshold,
                spread_max_px=max_px,
                min_alpha=min_alpha,
            )
            out_name = f"{base_tag}_{cat_name}__{cfg_tag}.png"
            _save_image(img, args.output_dir / out_name)


if __name__ == "__main__":
    main()
