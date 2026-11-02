"""Cluster the molecules of one or more datasets (``MoleculeClusteringBatchConfig``, §5.2.3).

    uv run python scripts/dataset_creation/cluster_molecules.py \
        --config configs/dataset_creation/cluster_molecules.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml

from remedi.data_handling.molecule_clustering import (
    MoleculeClusteringBatchConfig,
    cluster_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = pydantic_yaml.parse_yaml_file_as(
        MoleculeClusteringBatchConfig, arguments.config
    )
    for record in cluster_batch(config):
        print(
            f"{record.dataset_id}/{record.name}: {record.molecules} molecules, "
            f"{record.coarse_clusters} coarse, {record.fine_clusters} fine clusters"
        )


if __name__ == "__main__":
    main()
