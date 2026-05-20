"""BenchmarkBuildConfig: yaml round-trip + required roots + standardization toggles.

The script always builds the full panel, so both source roots are required
fields. These tests pin the yaml schema and the per-toggle round-trip.
"""

from __future__ import annotations

from pathlib import Path

import pydantic
import pydantic_yaml as pyd_yaml
import pytest

from threedscriptors.data_handling.dataset.tasks import ElementSet
from threedscriptors.data_handling.dataset_creation.build_config import (
    BenchmarkBuildConfig,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    MACE_OFF_ELEMENTS,
    MACE_POLAR_ELEMENTS,
    resolve_element_set,
)


def _minimal(tmp_path: Path, **overrides) -> BenchmarkBuildConfig:
    return BenchmarkBuildConfig(
        output_root=tmp_path / "out",
        moleculenet_raw_root=tmp_path / "raw",
        tdc_cache=tmp_path / "tdc",
        **overrides,
    )


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = _minimal(tmp_path)
    path = tmp_path / "build.yaml"
    pyd_yaml.to_yaml_file(path, cfg)
    loaded = pyd_yaml.parse_yaml_file_as(BenchmarkBuildConfig, path)
    assert loaded == cfg


def test_moleculenet_raw_root_required(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError, match="moleculenet_raw_root"):
        BenchmarkBuildConfig(
            output_root=tmp_path / "out", tdc_cache=tmp_path / "tdc"
        )


def test_tdc_cache_required(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError, match="tdc_cache"):
        BenchmarkBuildConfig(
            output_root=tmp_path / "out",
            moleculenet_raw_root=tmp_path / "raw",
        )


def test_standardization_defaults_and_roundtrip(tmp_path: Path) -> None:
    default = _minimal(tmp_path)
    assert default.strip_salts is True
    assert default.neutralize is True
    assert default.element_set is ElementSet.mace_off

    overridden = _minimal(
        tmp_path,
        strip_salts=False,
        neutralize=False,
        element_set=ElementSet.mace_polar,
    )
    path = tmp_path / "build.yaml"
    pyd_yaml.to_yaml_file(path, overridden)
    loaded = pyd_yaml.parse_yaml_file_as(BenchmarkBuildConfig, path)
    assert loaded == overridden
    assert loaded.element_set is ElementSet.mace_polar


def test_resolve_element_set_maps_to_concrete_sets() -> None:
    assert resolve_element_set(ElementSet.mace_off) is MACE_OFF_ELEMENTS
    assert resolve_element_set(ElementSet.mace_polar) is MACE_POLAR_ELEMENTS


def test_extra_field_forbidden(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError):
        BenchmarkBuildConfig(
            output_root=tmp_path / "out",
            moleculenet_raw_root=tmp_path / "raw",
            tdc_cache=tmp_path / "tdc",
            mystery_field="oops",  # type: ignore[call-arg]
        )
