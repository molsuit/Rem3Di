"""Shared fixtures for the ingestion-pipeline tests.

The integration tests run the real pipeline over the real pickles, which live
in the gitignored ``benchmark_data/raw/chiralcat`` of the Rem3Di checkout.
Tests that need the data are skipped when the pickles are absent. The full
build takes about a minute and a half, dominated by re-embedding in the repair
stage, so it runs once per session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chiralcat_dataset import PipelineConfig, build_dataset

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "pipeline.yaml"

data_present = pytest.mark.skipif(
    not (PipelineConfig.from_yaml(CONFIG_PATH).data_dir / "chiral_for_no.pkl").is_file(),
    reason="ChiralCat source pickles not present",
)


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    return PipelineConfig.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def fast_config() -> PipelineConfig:
    """The real config with the repair and rebuild stages switched off.

    Enough to exercise extraction, validation and the record plumbing without
    paying for re-embedding.
    """
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.repair.enabled = False
    config.organometallic.enabled = False
    return config


@pytest.fixture(scope="session")
def build(config: PipelineConfig):
    """Every stage over the real data, built once per session."""
    return build_dataset(config)
