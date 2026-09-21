"""Descriptor calculators + the content-addressed descriptor cache key (§7c).

EcfpCalculator only depends on ``dataset.get_smiles_per_structure()`` so it
duck-types fine over a small stub; RemediCalculator needs a real REM3DI
checkpoint, so we only test its config round-trip + cache key (real inference
is exercised by the end-to-end eval test).

The cache key carries the hash of the *data* and the hash of the *model*, not
just the dataset id and a user-chosen name. That is not cosmetic: keying on the
name alone served a matrix computed on an earlier ingest of the same endpoint,
and the row-count mismatch surfaced only as an ``IndexError`` when the split
mask was applied, four datasets into a panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydantic
import pydantic_yaml as pyd_yaml
import pytest
import yaml

from remedi.data_handling.bundle import (
    PROVENANCE_FILENAME,
    BundleOutputs,
    BundleProvenance,
    PreparerRecord,
    StructuresOutputRecord,
    TableOutputRecord,
    structures_identity,
)
from remedi.evaluation.benchmark.descriptors import (
    DescriptorConfig,
    EcfpCalculator,
    EcfpConfig,
    RemediConfig,
    compute_and_cache,
    descriptor_cache_path,
)


class _StubDataset:
    """Minimal duck for the SMILES path; the runner uses a real MoleculeDataset."""

    def __init__(self, smiles: list[str], path: Path) -> None:
        self._smiles = smiles
        self.path = path

    def get_smiles_per_structure(self) -> list[str]:
        return self._smiles


_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "O=C(O)c1ccccc1"]


def write_stub_dataset(directory: Path, structures_sha256: str | None) -> _StubDataset:
    """A directory that looks, to the cache key, like an ingested benchmark zarr.

    ``structures_sha256=None`` stands for a ``smiles``-stage bundle: no extxyz
    record, so the identity falls back to the table's content hash.
    """
    directory.mkdir(parents=True, exist_ok=True)
    outputs = BundleOutputs(
        **{
            "table.parquet": TableOutputRecord(
                file_sha256="f" * 64, content_sha256="c" * 64, rows=len(_SMILES)
            )
        }
    )
    if structures_sha256 is not None:
        outputs.structures_extxyz = StructuresOutputRecord(
            file_sha256=structures_sha256, frames=len(_SMILES)
        )
    provenance = BundleProvenance(
        dataset_id=directory.name,
        preparer=PreparerRecord(repo="tests", script="test_eval_descriptors.py"),
        outputs=outputs,
    )
    (directory / PROVENANCE_FILENAME).write_text(
        yaml.safe_dump(provenance.model_dump(mode="json", by_alias=True))
    )
    return _StubDataset(_SMILES, directory)


def test_ecfp_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=1024, name="ecfp4_1024")
    path = tmp_path / "ecfp.yaml"
    pyd_yaml.to_yaml_file(path, cfg)
    loaded = pyd_yaml.parse_yaml_file_as(EcfpConfig, path)
    assert loaded == cfg


def test_ecfp_calculator_shape_and_dtype(tmp_path: Path) -> None:
    calc = EcfpCalculator(fingerprint="ecfp", length=512)
    # The calculator itself never looks at the directory — only the cache does.
    X = calc.calculate(_StubDataset(_SMILES, tmp_path))  # type: ignore[arg-type]
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
    ds = write_stub_dataset(tmp_path / "toy", "a" * 64)
    cache_dir = tmp_path / "cache"

    X = compute_and_cache(cfg, ds, cache_dir, "toy")  # type: ignore[arg-type]
    cache_file = descriptor_cache_path(cfg, ds, cache_dir, "toy")  # type: ignore[arg-type]
    assert cache_file.exists()
    assert cache_file.name.startswith("toy__ecfp_128__")
    assert X.shape == (len(_SMILES), 128)

    # Second call with the identical config hits the cache.
    X2 = compute_and_cache(cfg, ds, cache_dir, "toy")  # type: ignore[arg-type]
    np.testing.assert_array_equal(X, X2)
    assert list(cache_dir.iterdir()) == [cache_file]


def test_compute_and_cache_uses_dataset_id_namespace(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=64, name="ecfp")
    esol = write_stub_dataset(tmp_path / "esol", "a" * 64)
    lipo = write_stub_dataset(tmp_path / "lipo", "b" * 64)

    X_a = compute_and_cache(cfg, esol, tmp_path / "cache", "esol")  # type: ignore[arg-type]
    X_b = compute_and_cache(cfg, lipo, tmp_path / "cache", "lipo")  # type: ignore[arg-type]

    assert len(list((tmp_path / "cache").iterdir())) == 2
    np.testing.assert_array_equal(X_a, X_b)  # same SMILES -> same descriptors


# ------------------------------------------------- the two halves of the key


def test_same_dataset_id_different_structures_do_not_share_a_cache_file(
    tmp_path: Path,
) -> None:
    """The defect this key exists to prevent: a re-ingest under the same name."""
    cfg = EcfpConfig(length=64, name="ecfp")
    first = write_stub_dataset(tmp_path / "first" / "HIA_Hou", "a" * 64)
    second = write_stub_dataset(tmp_path / "second" / "HIA_Hou", "b" * 64)
    cache_dir = tmp_path / "cache"

    first_path = descriptor_cache_path(cfg, first, cache_dir, "HIA_Hou")  # type: ignore[arg-type]
    second_path = descriptor_cache_path(cfg, second, cache_dir, "HIA_Hou")  # type: ignore[arg-type]

    assert first_path != second_path
    compute_and_cache(cfg, first, cache_dir, "HIA_Hou")  # type: ignore[arg-type]
    compute_and_cache(cfg, second, cache_dir, "HIA_Hou")  # type: ignore[arg-type]
    assert first_path.exists() and second_path.exists()


def test_a_reused_descriptor_name_with_different_settings_splits_the_cache(
    tmp_path: Path,
) -> None:
    ds = write_stub_dataset(tmp_path / "toy", "a" * 64)
    short = EcfpConfig(length=64, name="ecfp")
    long = EcfpConfig(length=128, name="ecfp")  # same label, different matrix

    short_path = descriptor_cache_path(short, ds, tmp_path, "toy")  # type: ignore[arg-type]
    long_path = descriptor_cache_path(long, ds, tmp_path, "toy")  # type: ignore[arg-type]

    assert short_path != long_path
    assert short.identity() != long.identity()


def test_the_structures_hash_falls_back_to_the_table_content_hash(
    tmp_path: Path,
) -> None:
    ds = write_stub_dataset(tmp_path / "smiles_stage", None)

    assert structures_identity(ds.path) == "c" * 64


def test_a_dataset_without_provenance_names_the_problem(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=64, name="ecfp")
    ds = _StubDataset(_SMILES, tmp_path / "no_provenance")
    ds.path.mkdir()

    with pytest.raises(FileNotFoundError, match=r"provenance\.yaml"):
        compute_and_cache(cfg, ds, tmp_path / "cache", "toy")  # type: ignore[arg-type]


def test_the_remedi_identity_is_the_checkpoint_hash_and_is_memoised(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "encoder.pth").write_bytes(b"weights-v1")
    cfg = RemediConfig(name="remedi", model_dir=model_dir)

    identity = cfg.identity()

    assert cfg.checkpoint_sha256 == identity
    # Knobs that do not change the matrix do not change the identity.
    assert (
        RemediConfig(
            name="remedi", model_dir=model_dir, batch_size=8, device="cpu"
        ).identity()
        == identity
    )
    # Different weights under the same name do.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "encoder.pth").write_bytes(b"weights-v2")
    assert RemediConfig(name="remedi", model_dir=other_dir).identity() != identity


def test_a_remedi_config_without_weights_names_the_problem(tmp_path: Path) -> None:
    cfg = RemediConfig(name="remedi", model_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match=r"no \.pth checkpoint"):
        cfg.identity()
