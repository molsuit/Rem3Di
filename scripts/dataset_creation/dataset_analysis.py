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

import pydantic_yaml as pyd_yaml

from threedscriptors.configuration.dataset_analysis_config import (
    MoleculeDatasetAnalysisConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import MoleculeDatasetAnalysis


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
    return pyd_yaml.parse_yaml_file_as(MoleculeDatasetAnalysisConfig, file=path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Opening dataset at %s", args.dataset_dir)
    dataset = MoleculeDataset.open_existing_dataset_from_dir(args.dataset_dir)

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
