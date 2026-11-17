"""Every evaluation and dataset-analysis yaml under configs/ parses."""

from __future__ import annotations

from pathlib import Path

import pytest

from remedi.embedding.pipeline import EmbedAndEvaluateConfig
from remedi_data.yaml_io import read_model
from remedi_eval.dataset_analysis.config import DatasetComparisonConfig
from remedi_eval.framework import EvalManifest

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    "path", sorted((CONFIGS / "evaluation").glob("*.yaml")), ids=lambda path: path.name
)
def test_the_evaluation_manifests_parse(path: Path) -> None:
    read_model(path, EvalManifest)


@pytest.mark.parametrize(
    "path",
    sorted((CONFIGS / "evaluation" / "descriptor_analysis").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_the_descriptor_analysis_configs_parse(path: Path) -> None:
    read_model(path, EmbedAndEvaluateConfig)


def test_the_dataset_comparison_config_parses() -> None:
    read_model(
        CONFIGS / "dataset_creation" / "dataset_comparison_dev.yaml",
        DatasetComparisonConfig,
    )
