"""Build datasets from SMILES bundles (conformers -> zarr) and verify every dataset.

One YAML (``DatasetBuildConfig``) describes one run. Each bundle is one task, so
a dataset that fails is recorded in ``status.yaml`` while the rest still finish.

Usage::

    uv run python scripts/dataset_creation/build_datasets.py \
        --config configs/dataset_creation/build_datasets_template.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from remedi_data.build import DatasetBuildConfig, build_datasets
from remedi_data.logging_setup import setup_logging
from remedi_data.yaml_io import read_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    setup_logging()
    config = read_model(arguments.config, DatasetBuildConfig)
    report = build_datasets(config)
    return 1 if report.n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
