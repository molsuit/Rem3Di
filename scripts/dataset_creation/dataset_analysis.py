"""Run MoleculeDatasetAnalysis on an existing on-disk dataset.

Usage:
    uv run scripts/dataset_creation/dataset_analysis.py \
        --dataset-dir /path/to/dataset \
        --output-dir /path/to/output \
        [--config /path/to/analysis_config.yaml]
"""

import argparse
import logging
from pathlib import Path

from remedi_data.dataset import Dataset
from remedi_data.logging_setup import setup_logging
from remedi_data.yaml_io import read_model
from remedi_eval.dataset_analysis.analysis import MoleculeDatasetAnalysis
from remedi_eval.dataset_analysis.config import (
    MoleculeDatasetAnalysisConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="Path to the on-disk dataset."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Where to write analysis output."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config (MoleculeDatasetAnalysisConfig). If omitted,"
            " defaults are used."
        ),
    )
    return parser.parse_args()


def load_config(path: Path | None) -> MoleculeDatasetAnalysisConfig:
    if path is None:
        return MoleculeDatasetAnalysisConfig()
    return read_model(path, MoleculeDatasetAnalysisConfig)


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Opening dataset at %s", args.dataset_dir)
    dataset = Dataset.open(args.dataset_dir)

    analysis = MoleculeDatasetAnalysis(dataset, config=config)
    summary = analysis.run()
    analysis.output(args.output_dir)

    logging.info(
        "Analysis complete: %d structures / %d molecules. Summary written to %s",
        summary.n_structures,
        summary.n_molecules,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
