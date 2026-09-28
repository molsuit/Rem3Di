"""Configuration models: discriminated unions, path resolution, validation."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from chiralcat_dataset.config import (
    CorrectionSet,
    DeleteCorrection,
    FixedLabelSource,
    PipelineConfig,
    RelabelCorrection,
    TypedSource,
)

MINIMAL = {
    "extraction": {
        "data_dir": "data",
        "sources": [
            {"kind": "typed", "path": "typed.pkl"},
            {"kind": "fixed_label", "path": "fixed.pkl", "label": "axial"},
        ],
    }
}


def _write(tmp_path, payload, name="pipeline.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def test_source_union_discriminates_on_kind(tmp_path):
    config = PipelineConfig.from_yaml(_write(tmp_path, MINIMAL))
    typed, fixed = config.extraction.sources
    assert isinstance(typed, TypedSource)
    assert isinstance(fixed, FixedLabelSource)
    assert typed.drop_types == ["unknown"]
    assert fixed.label == "axial"


def test_unknown_source_kind_is_rejected(tmp_path):
    payload = {"extraction": {"sources": [{"kind": "nonsense", "path": "x.pkl"}]}}
    with pytest.raises(ValidationError):
        PipelineConfig.from_yaml(_write(tmp_path, payload))


def test_fixed_label_source_with_unknown_class_is_rejected(tmp_path):
    payload = {
        "extraction": {
            "sources": [{"kind": "fixed_label", "path": "x.pkl", "label": "sideways"}]
        }
    }
    with pytest.raises(ValueError, match="Unknown chiral class"):
        PipelineConfig.from_yaml(_write(tmp_path, payload))


def test_paths_resolve_against_the_config_file_not_the_cwd(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    config = PipelineConfig.from_yaml(_write(nested, MINIMAL))
    assert config.data_dir == (nested / "data").resolve()
    assert config.output_dir == (nested / "output").resolve()


def test_correction_union_discriminates_on_kind(tmp_path):
    path = tmp_path / "corrections.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "corrections": [
                    {"kind": "relabel", "smiles": "CCO", "to_class": "achiral"},
                    {"kind": "delete", "smiles": "CCC", "reason": "ambiguous"},
                ]
            }
        )
    )
    corrections = CorrectionSet.from_yaml(path).corrections
    assert isinstance(corrections[0], RelabelCorrection)
    assert isinstance(corrections[1], DeleteCorrection)


def test_missing_corrections_file_yields_empty_set(tmp_path):
    payload = dict(MINIMAL)
    payload["validation"] = {"corrections_file": None}
    config = PipelineConfig.from_yaml(_write(tmp_path, payload))
    assert config.corrections_path is None
    assert config.load_corrections().corrections == []


def test_defaults_cover_every_stage(tmp_path):
    config = PipelineConfig.from_yaml(_write(tmp_path, MINIMAL))
    assert config.repair.enabled is True
    assert config.organometallic.target_class == "planar"
    assert config.validation.on_uncovered_mislabel == "warn"
    assert config.output.dataset_index == "dataset.csv"
