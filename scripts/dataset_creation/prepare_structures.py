"""Prepare structure corpora (PCQM4Mv2, GEOM-Drugs, tmQM, ...) into datasets.

One YAML (``StructurePreparationConfig``) describes one run; each source is one
task, and each source runs in restartable shards: rerunning after a crash or a
timeout redoes only the unfinished shards.

Usage::

    uv run python scripts/dataset_creation/prepare_structures.py \
        --config configs/dataset_creation/prepare_structures.yaml [--only tmqm]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from remedi_data.structures import (
    StructurePreparationConfig,
    prepare_structure_datasets,
)
from remedi_data.yaml_io import read_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None, metavar="DATASET_ID")
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = read_model(arguments.config, StructurePreparationConfig)
    report = prepare_structure_datasets(config, only=arguments.only)
    return 1 if report.n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
