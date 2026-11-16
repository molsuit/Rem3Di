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

from remedi_data.logging_setup import setup_logging
from remedi_data.yaml_io import read_model
from remedi_eval.dataset_analysis.comparison import DatasetComparison
from remedi_eval.dataset_analysis.config import (
    DatasetComparisonConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = read_model(args.config, DatasetComparisonConfig)
    comparison = DatasetComparison(config)
    comparison.run()
    out = comparison.output()
    logging.getLogger(__name__).info(
        "Wrote %d artifacts to %s", len(comparison.results), out
    )


if __name__ == "__main__":
    main()
