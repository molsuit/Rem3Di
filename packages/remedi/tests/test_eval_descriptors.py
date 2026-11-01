"""The ECFP descriptor cache: embeddings pairs keyed by data and config identity.

A cached matrix is reused only when its ``structures_sha256`` and descriptor
identity match the request. Keying on the name alone once served a matrix
computed on an earlier build of the same endpoint; the row-count mismatch
surfaced only as an ``IndexError`` four datasets into a panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from remedi.embedding import RemediConfig
from remedi_data.bundle import (
    PROVENANCE_FILENAME,
    BundleOutputs,
    BundleProvenance,
    PreparerRecord,
    TableOutputRecord,
    structures_identity,
)
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors import (
    EcfpConfig,
    compute_and_cache,
    embeddings_paths,
    read_metadata,
)


class _StubDataset:
    """Minimal duck for the SMILES path; the runner uses a real MoleculeDataset."""

    def __init__(self, smiles: list[str], path: Path) -> None:
        self._smiles = smiles
        self.path = path
        self.structure_ids = np.arange(len(smiles), dtype=np.int64)

    def get_smiles_per_structure(self) -> list[str]:
        return self._smiles


_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "O=C(O)c1ccccc1"]


def stub_dataset(directory: Path) -> MoleculeDataset:
    return cast(MoleculeDataset, _StubDataset(_SMILES, directory))


def write_stub_dataset(
    directory: Path, structures_sha256: str | None
) -> MoleculeDataset:
    """A directory that looks, to the cache key, like a dataset.

    ``structures_sha256=None`` stands for a bundle: no structures, so the
    identity falls back to the table's content hash.
    """
    directory.mkdir(parents=True, exist_ok=True)
    outputs = BundleOutputs(
        table_parquet=TableOutputRecord(
            file_sha256="f" * 64, content_sha256="c" * 64, rows=len(_SMILES)
        ),
        structures_sha256=structures_sha256,
    )
    provenance = BundleProvenance(
        dataset_id=directory.name,
        preparer=PreparerRecord(repo="tests", script="test_eval_descriptors.py"),
        outputs=outputs,
    )
    (directory / PROVENANCE_FILENAME).write_text(
        yaml.safe_dump(provenance.model_dump(mode="json", by_alias=True))
    )
    return stub_dataset(directory)


def test_compute_and_cache_writes_then_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = EcfpConfig(length=128, name="ecfp_128")
    ds = write_stub_dataset(tmp_path / "toy", "a" * 64)
    cache_dir = tmp_path / "cache"

    X = compute_and_cache(cfg, ds, cache_dir, "toy")
    npz_path, yaml_path = embeddings_paths(cache_dir, "toy", "ecfp_128")
    assert npz_path.exists() and yaml_path.exists()
    assert X.shape == (len(_SMILES), 128)
    assert X.dtype == np.float32
    metadata = read_metadata(cache_dir, "toy", "ecfp_128")
    assert metadata is not None
    assert metadata.descriptor_identity == cfg.identity()
    assert metadata.structures_sha256 == "a" * 64

    # Second call with the identical config hits the cache: nothing is computed.
    monkeypatch.setattr(EcfpConfig, "build", lambda self: pytest.fail("recomputed"))
    X2 = compute_and_cache(cfg, ds, cache_dir, "toy")
    np.testing.assert_array_equal(X, X2)
    assert sorted(path.name for path in (cache_dir / "toy").iterdir()) == [
        "ecfp_128.npz",
        "ecfp_128.yaml",
    ]


# ------------------------------------------------- the two halves of the key


@pytest.mark.parametrize(
    ("second_structures_sha256", "length"),
    [
        # The defect this key exists to prevent: a rebuild under the same name.
        ("b" * 64, 64),
        # A reused descriptor name with different settings is another matrix.
        ("a" * 64, 128),
    ],
    ids=["structures", "descriptor_settings"],
)
def test_a_mismatched_cache_entry_is_recomputed_and_overwritten(
    tmp_path: Path, second_structures_sha256: str, length: int
) -> None:
    cache_dir = tmp_path / "cache"
    first = write_stub_dataset(tmp_path / "first" / "HIA_Hou", "a" * 64)
    second = write_stub_dataset(
        tmp_path / "second" / "HIA_Hou", second_structures_sha256
    )
    compute_and_cache(EcfpConfig(length=64, name="ecfp"), first, cache_dir, "HIA_Hou")

    second_cfg = EcfpConfig(length=length, name="ecfp")
    X = compute_and_cache(second_cfg, second, cache_dir, "HIA_Hou")

    assert X.shape == (len(_SMILES), length)
    metadata = read_metadata(cache_dir, "HIA_Hou", "ecfp")
    assert metadata is not None
    assert metadata.structures_sha256 == second_structures_sha256
    assert metadata.descriptor_identity == second_cfg.identity()


def test_dataset_ids_do_not_share_a_cache_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cfg = EcfpConfig(length=64, name="ecfp")
    for dataset_id in ["lipo", "HIA_Hou"]:
        compute_and_cache(
            cfg,
            write_stub_dataset(tmp_path / dataset_id, "a" * 64),
            cache_dir,
            dataset_id,
        )
    assert sorted(path.name for path in cache_dir.iterdir()) == ["HIA_Hou", "lipo"]


def test_the_structures_hash_falls_back_to_the_table_content_hash(
    tmp_path: Path,
) -> None:
    ds = write_stub_dataset(tmp_path / "bundle", None)

    assert structures_identity(ds.path) == "c" * 64


def test_a_dataset_without_provenance_names_the_problem(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=64, name="ecfp")
    ds = stub_dataset(tmp_path / "no_provenance")
    ds.path.mkdir()

    with pytest.raises(FileNotFoundError, match=r"provenance\.yaml"):
        compute_and_cache(cfg, ds, tmp_path / "cache", "toy")


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
    (other_dir / "empty").mkdir()

    # A model directory without weights names the problem.
    with pytest.raises(FileNotFoundError, match=r"no \.pth checkpoint"):
        RemediConfig(name="remedi", model_dir=other_dir / "empty").identity()
