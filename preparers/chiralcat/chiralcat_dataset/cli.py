"""Build the ChiralCat chirality dataset from the source pickles.

Runs the whole ingestion pipeline (extract -> validate -> repair -> rebuild) and
writes two datasets to the configured output directory:

    dataset.csv / dataset.extxyz     structures usable for training
    rejected.csv / rejected.extxyz   structures that were erroneous and not fixed
    run_report.json                  per-stage counts, corrections, config

Run from the Rem3Di repository root::

    uv run --no-sync build-chiralcat-dataset [--config pipeline.yaml] [--no-write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import build_dataset, write_outputs
from .validation import UncoveredMislabelError

#: The committed pipeline config beside this package.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "pipeline.yaml"


def _print_summary(result, paths) -> None:
    print(f"\ndataset: {len(result.structures)} structures")
    for class_name, count in result.class_counts.items():
        print(f"    {class_name:<9} {count:>6}")

    quality: dict[str, int] = {}
    for structure in result.structures:
        quality[structure.geometry_quality] = (
            quality.get(structure.geometry_quality, 0) + 1
        )
    print(
        "  geometry quality: "
        + ", ".join(f"{key}={value}" for key, value in sorted(quality.items()))
    )

    broken = result.rejected_broken
    filtered = result.rejected_filtered
    print(
        f"\nrejected: {len(result.rejected)} "
        f"({len(broken)} broken, {len(filtered)} deliberately filtered)"
    )
    reasons: dict[str, int] = {}
    for record in broken:
        key = f"{record.stage}:{record.reason}"
        reasons[key] = reasons.get(key, 0) + 1
    for key, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"    {key:<28} {count:>5}")

    applied = sum(1 for correction in result.corrections if correction.matched)
    unmatched = sum(1 for correction in result.corrections if not correction.matched)
    print(f"\nlabel corrections: {applied} applied, {unmatched} unmatched")
    if result.uncovered_mislabels:
        print(
            f"WARNING: {len(result.uncovered_mislabels)} central molecule(s) have no "
            "3D stereocentre and are not covered by corrections.yaml:"
        )
        for smiles in result.uncovered_mislabels[:10]:
            print(f"    {smiles}")
        if len(result.uncovered_mislabels) > 10:
            print(f"    ... and {len(result.uncovered_mislabels) - 10} more")

    print("\nwrote:")
    for name, path in paths.items():
        print(f"    {name:<16} {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"pipeline configuration YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run the pipeline and print the summary without writing any file",
    )
    args = parser.parse_args(argv)

    try:
        config = PipelineConfig.from_yaml(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: could not load config {args.config}: {exc}", file=sys.stderr)
        return 2

    try:
        result = build_dataset(
            config, progress=lambda message: print(message, flush=True)
        )
    except UncoveredMislabelError as exc:
        print(f"error: label validation failed\n{exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: pipeline failed: {exc}", file=sys.stderr)
        return 1

    paths = {} if args.no_write else write_outputs(config, result)
    _print_summary(result, paths)
    return 0
