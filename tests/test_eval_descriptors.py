"""The content-addressed descriptor cache key (§7c), over a stub dataset.

The cache key carries the hash of the *data* and the hash of the *model*, not
just the dataset id and a user-chosen name. That is not cosmetic: keying on the
name alone served a matrix computed on an earlier build of the same endpoint,
and the row-count mismatch surfaced only as an ``IndexError`` when the split
mask was applied, four datasets into a panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from remedi.data_handling.bundle import (
    PROVENANCE_FILENAME,
    BundleOutputs,
    BundleProvenance,
    PreparerRecord,
    TableOutputRecord,
    structures_identity,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.benchmark.descriptors import (
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


def test_compute_and_cache_writes_then_short_circuits(tmp_path: Path) -> None:
    cfg = EcfpConfig(length=128, name="ecfp_128")
    ds = write_stub_dataset(tmp_path / "toy", "a" * 64)
    cache_dir = tmp_path / "cache"

    X = compute_and_cache(cfg, ds, cache_dir, "toy")
    cache_file = descriptor_cache_path(cfg, ds, cache_dir, "toy")
    assert cache_file.exists()
    assert cache_file.name.startswith("toy__ecfp_128__")
    assert X.shape == (len(_SMILES), 128)
    assert X.dtype == np.float32

    # Second call with the identical config hits the cache.
    X2 = compute_and_cache(cfg, ds, cache_dir, "toy")
    np.testing.assert_array_equal(X, X2)
    assert list(cache_dir.iterdir()) == [cache_file]


# ------------------------------------------------- the two halves of the key


@pytest.mark.parametrize(
    ("second_directory", "second_structures_sha256", "second_dataset_id", "length"),
    [
        # Another dataset id is another namespace.
        ("lipo", "a" * 64, "lipo", 64),
        # The defect this key exists to prevent: a rebuild under the same name.
        ("second/HIA_Hou", "b" * 64, "HIA_Hou", 64),
        # A reused descriptor name with different settings is another matrix.
        ("first/HIA_Hou", "a" * 64, "HIA_Hou", 128),
    ],
    ids=["dataset_id", "structures", "descriptor_settings"],
)
def test_the_cache_key_separates(
    tmp_path: Path,
    second_directory: str,
    second_structures_sha256: str,
    second_dataset_id: str,
    length: int,
) -> None:
    cache_dir = tmp_path / "cache"
    first_cfg = EcfpConfig(length=64, name="ecfp")
    second_cfg = EcfpConfig(length=length, name="ecfp")
    first = write_stub_dataset(tmp_path / "first" / "HIA_Hou", "a" * 64)
    second = write_stub_dataset(tmp_path / second_directory, second_structures_sha256)

    for cfg, ds, dataset_id in [
        (first_cfg, first, "HIA_Hou"),
        (second_cfg, second, second_dataset_id),
    ]:
        compute_and_cache(cfg, ds, cache_dir, dataset_id)

    assert len(list(cache_dir.iterdir())) == 2


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
