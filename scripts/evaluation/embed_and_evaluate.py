"""Embed every panel dataset with a Rem3Di checkpoint, then run the eval manifest.

The config holds ``embedding`` (a :class:`RemediConfig`) and ``manifest`` (an
eval manifest whose ``model`` is ``descriptor_kind: precomputed`` with the same
``name``). Datasets whose embeddings already match are not re-embedded. Run it
as one GPU job.

Usage::

    uv run python scripts/evaluation/embed_and_evaluate.py --config <config>.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from remedi.embedding import EmbedAndEvaluateConfig, embed_and_evaluate

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = pyd_yaml.parse_yaml_file_as(EmbedAndEvaluateConfig, args.config)
    embedding_report, run_report = embed_and_evaluate(config)
    logger.info(
        "embedded %d, up to date %d, failed %d; eval %d/%d task(s) ok -> %s",
        len(embedding_report.written),
        len(embedding_report.skipped),
        len(embedding_report.failed),
        run_report.n_tasks - run_report.n_failed,
        run_report.n_tasks,
        config.manifest.output_root,
    )
    if embedding_report.failed or run_report.n_failed:
        raise SystemExit(
            "some datasets or tasks failed; see embedding_status.yaml and status.yaml"
        )


if __name__ == "__main__":
    main()
