"""Build datasets from SMILES bundles (conformers -> zarr) and verify every dataset.

One YAML (``DatasetBuildConfig``) describes one run. Each bundle is one task, so
a dataset that fails is recorded in ``status.yaml`` while the rest still finish.

Usage::

    uv run python scripts/dataset_creation/build_datasets.py \
        --config configs/dataset_creation/build_datasets_template.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pydantic_yaml

from remedi.data_handling.dataset_build import DatasetBuildConfig, build_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = pydantic_yaml.parse_yaml_file_as(DatasetBuildConfig, arguments.config)
    report = build_datasets(config)
    return 1 if report.n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
