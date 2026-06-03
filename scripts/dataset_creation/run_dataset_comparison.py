"""Compare a set of role-tagged molecular datasets in scaffold/Tanimoto space.

Loads a yaml of `DatasetComparisonConfig`, runs `DatasetComparison`, and writes
plots + a summary yaml to ``output_dir``.

Usage::

    uv run python scripts/dataset_creation/run_dataset_comparison.py \\
        --config configs/dataset_creation/dataset_comparison_dev.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from threedscriptors.configuration.dataset_comparison_config import (
    DatasetComparisonConfig,
)
from threedscriptors.data_handling.dataset_comparison import DatasetComparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = pyd_yaml.parse_yaml_file_as(DatasetComparisonConfig, file=args.config)
    comparison = DatasetComparison(config)
    comparison.run()
    out = comparison.output()
    logging.getLogger(__name__).info("Wrote %d artifacts to %s", len(comparison.results), out)


if __name__ == "__main__":
    main()
