"""Make the selections of one yaml (``SelectionBatchConfig``, BENCHMARK_DATA_FORMAT.md §5.2.3).

    uv run python scripts/dataset_creation/select_structures.py \
        --config configs/dataset_creation/select_structures.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml

from remedi_data.selection import SelectionBatchConfig, select_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = pydantic_yaml.parse_yaml_file_as(SelectionBatchConfig, arguments.config)
    for record in select_batch(config):
        print(
            f"{record.dataset_id}/{record.name}: {record.structures} structures, "
            f"{record.molecules} molecules, {record.atoms} atoms"
        )


if __name__ == "__main__":
    main()
