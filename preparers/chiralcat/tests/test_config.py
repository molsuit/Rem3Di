"""Configuration: path resolution, class validation and the curation defaults."""

from __future__ import annotations

import pytest
import yaml

from chiralcat_dataset.config import PipelineConfig, TypedSource

MINIMAL = {
    "extraction": {
        "data_dir": "data",
        "sources": [
            {"kind": "typed", "path": "typed.pkl"},
            {"kind": "fixed_label", "path": "fixed.pkl", "label": "axial"},
        ],
    }
}


def _load(directory, payload) -> PipelineConfig:
    path = directory / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload))
    return PipelineConfig.from_yaml(path)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "extraction": {
                "sources": [{"kind": "fixed_label", "path": "x.pkl", "label": "sideways"}]
            }
        },
        {**MINIMAL, "organometallic": {"target_class": "sideways"}},
    ],
)
def test_an_unknown_class_anywhere_in_the_config_is_rejected(tmp_path, payload):
    with pytest.raises(ValueError, match="Unknown chiral class"):
        _load(tmp_path, payload)


def test_paths_resolve_against_the_config_file_not_the_cwd(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    config = _load(nested, MINIMAL)
    assert config.data_dir == (nested / "data").resolve()
    assert config.output_dir == (nested / "output").resolve()
    assert config.corrections_path == (nested / "corrections.yaml").resolve()


def test_defaults_encode_the_curation_policy(tmp_path):
    config = _load(tmp_path, MINIMAL)
    typed_source = config.extraction.sources[0]
    assert isinstance(typed_source, TypedSource)
    assert typed_source.drop_types == ["unknown"]
    assert config.repair.enabled is True
    assert config.organometallic.target_class == "planar"
    assert config.validation.on_uncovered_mislabel == "warn"
    assert config.validation.audit_central_stereo is True
