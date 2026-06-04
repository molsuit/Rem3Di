"""Regenerate figures for a finished eval run from its saved data artifacts.

Because tasks emit pure-data artifacts (``results.csv`` etc.), plots can be
re-rendered offline — change a plotter and rebuild every figure without
re-running the (GPU) eval. Figures are written under ``<run-dir>/plots/``.

Usage::

    uv run python scripts/evaluation/replot.py --run-dir <eval_base>/<model>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Import for the side effect of registering the built-in plotters.
import threedscriptors.evaluation.framework.builtin_plotters  # noqa: F401
from threedscriptors.evaluation.framework.plotting import render
from threedscriptors.evaluation.results import TableResult

logger = logging.getLogger(__name__)

# Maps a relative artifact path under the run dir to a registered plotter kind.
_ARTIFACT_PLOTTERS = {
    "benchmark/results.csv": "benchmark_results",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    plots_dir = run_dir / "plots"
    total = 0
    for rel, kind in _ARTIFACT_PLOTTERS.items():
        path = run_dir / rel
        if not path.exists():
            logger.info("skip %s (no %s)", kind, rel)
            continue
        figs = render(kind, TableResult.load(path), plots_dir)
        logger.info("%s -> %d figure(s)", rel, len(figs))
        total += len(figs)

    if total == 0:
        logger.warning("no figures produced for %s", run_dir)
    else:
        logger.info("wrote %d figure(s) to %s", total, plots_dir)


if __name__ == "__main__":
    main()
