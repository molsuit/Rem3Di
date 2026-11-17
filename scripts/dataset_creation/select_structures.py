"""Make the selections of one yaml (``SelectionBatchConfig``, BENCHMARK_DATA_FORMAT.md §5.2.3).

    uv run python scripts/dataset_creation/select_structures.py \
        --config configs/dataset_creation/select_structures.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from remedi_data.logging_setup import setup_logging
from remedi_data.selection import SelectionBatchConfig, select_batch
from remedi_data.yaml_io import read_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    setup_logging()
    config = read_model(arguments.config, SelectionBatchConfig)
    for record in select_batch(config):
        print(
            f"{record.dataset_id}/{record.name}: {record.structures} structures, "
            f"{record.molecules} molecules, {record.atoms} atoms"
        )


if __name__ == "__main__":
    main()
