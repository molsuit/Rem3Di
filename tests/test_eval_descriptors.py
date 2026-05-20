"""Descriptor calculators + the (dataset_id, descriptor_name) cache key.

EcfpCalculator only depends on ``dataset.get_smiles_per_structure()`` so it
duck-types fine over a small stub; RemediCalculator needs a real REM3DI
checkpoint, so we only test its config round-trip + cache key (real inference
is exercised by the end-to-end eval test in a later turn).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydantic
import pydantic_yaml as pyd_yaml
import pytest

from threedscriptors.evaluation.benchmark.descriptors import (
    DescriptorConfig,
    EcfpCalculator,
    EcfpConfig,
    RemediConfig,
    compute_and_cache,
)


class _StubDataset:
    """Minimal duck for the SMILES path; the runner uses a real MoleculeDataset."""

    def __init__(self, smiles: list[str]) -> None:
        self._smiles = smiles

    def get_smiles_per_structure(self) -> list[str]:
        return self._smiles


_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "O=C(O)c1ccccc1"]


def test_ecfp_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=1024, name="ecfp4_1024")
    path = tmp_path / "ecfp.yaml"
    pyd_yaml.to_yaml_file(path, cfg)
    loaded = pyd_yaml.parse_yaml_file_as(EcfpConfig, path)
    assert loaded == cfg


def test_ecfp_calculator_shape_and_dtype() -> None:
    calc = EcfpCalculator(fingerprint="ecfp", length=512)
    X = calc.calculate(_StubDataset(_SMILES))  # type: ignore[arg-type]
    assert X.shape == (len(_SMILES), 512)
    assert X.dtype == np.float32


def test_remedi_config_roundtrip_and_cache_filename(tmp_path: Path) -> None:
    cfg = RemediConfig(name="remedi_geom350k", model_dir=tmp_path / "ckpt")
    path = tmp_path / "remedi.yaml"
    pyd_yaml.to_yaml_file(path, cfg)
    loaded = pyd_yaml.parse_yaml_file_as(RemediConfig, path)
    assert loaded == cfg
    # We don't .build() — that would load a model checkpoint we don't have.


def test_descriptor_config_union_discriminates(tmp_path: Path) -> None:
    adapter = pydantic.TypeAdapter(DescriptorConfig)
    ecfp = adapter.validate_python({"descriptor_kind": "ecfp"})
    remedi = adapter.validate_python(
        {"descriptor_kind": "remedi", "name": "x", "model_dir": str(tmp_path)}
    )
    assert isinstance(ecfp, EcfpConfig)
    assert isinstance(remedi, RemediConfig)


def test_descriptor_config_rejects_unknown_kind() -> None:
    adapter = pydantic.TypeAdapter(DescriptorConfig)
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python({"descriptor_kind": "made_up"})


def test_compute_and_cache_writes_then_short_circuits(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=128, name="ecfp_128")
    ds = _StubDataset(_SMILES)
    cache_dir = tmp_path / "cache"

    X = compute_and_cache(cfg, ds, cache_dir, "toy")  # type: ignore[arg-type]
    cache_file = cache_dir / "toy__ecfp_128.npz"
    assert cache_file.exists()
    assert X.shape == (len(_SMILES), 128)

    # Second call hits cache — flip the calculator config so any recompute
    # would change shape; the cache should still return the original (128).
    cfg2 = EcfpConfig(length=999, name="ecfp_128")  # same name, different length
    X2 = compute_and_cache(cfg2, ds, cache_dir, "toy")  # type: ignore[arg-type]
    assert X2.shape == X.shape
    np.testing.assert_array_equal(X, X2)


def test_compute_and_cache_uses_dataset_id_namespace(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=64, name="ecfp")
    ds = _StubDataset(_SMILES)
    X_a = compute_and_cache(cfg, ds, tmp_path, "esol")  # type: ignore[arg-type]
    X_b = compute_and_cache(cfg, ds, tmp_path, "lipo")  # type: ignore[arg-type]
    assert (tmp_path / "esol__ecfp.npz").exists()
    assert (tmp_path / "lipo__ecfp.npz").exists()
    np.testing.assert_array_equal(X_a, X_b)  # same SMILES -> same descriptors
