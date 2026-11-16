"""The ECFP descriptor cache: embeddings pairs keyed by data and config identity.

A cached matrix is reused only when its ``structures_sha256`` and descriptor
identity match the request. Keying on the name alone once served a matrix
computed on an earlier build of the same endpoint; the row-count mismatch
surfaced only as an ``IndexError`` four datasets into a panel.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from remedi_data.bundle import (
    PROVENANCE_FILENAME,
    BundleOutputs,
    BundleProvenance,
    PreparerRecord,
    TableOutputRecord,
)
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors import (
    EcfpConfig,
    embeddings_paths,
    ensure_embeddings,
    producer_record,
    read_embeddings,
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


def test_the_matrix_is_written_then_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = EcfpConfig(fingerprint_size=128, name="ecfp_128")
    ds = write_stub_dataset(tmp_path / "toy", "a" * 64)
    cache_dir = tmp_path / "cache"

    X = cfg.load_matrix(ds, "toy", cache_dir)
    npz_path, yaml_path = embeddings_paths(cache_dir, "toy", "ecfp_128")
    assert npz_path.exists() and yaml_path.exists()
    assert X.shape == (len(_SMILES), 128)
    assert X.dtype == np.float32
    metadata = read_metadata(cache_dir, "toy", "ecfp_128")
    assert metadata is not None
    assert metadata.descriptor_identity == cfg.identity()
    assert metadata.structures_sha256 == "a" * 64

    # Second call with the identical config hits the cache: nothing is computed.
    monkeypatch.setattr(
        EcfpConfig, "calculate", lambda self, dataset: pytest.fail("recomputed")
    )
    X2 = cfg.load_matrix(ds, "toy", cache_dir)
    np.testing.assert_array_equal(X, X2)
    visible = [
        path.name
        for path in (cache_dir / "toy").iterdir()
        if not path.name.startswith(".")
    ]
    assert sorted(visible) == ["ecfp_128.npz", "ecfp_128.yaml"]
    assert not list((cache_dir / "toy").glob("*partial*"))


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
    EcfpConfig(fingerprint_size=64, name="ecfp").load_matrix(
        first, "HIA_Hou", cache_dir
    )

    second_cfg = EcfpConfig(fingerprint_size=length, name="ecfp")
    X = second_cfg.load_matrix(second, "HIA_Hou", cache_dir)

    assert X.shape == (len(_SMILES), length)
    metadata = read_metadata(cache_dir, "HIA_Hou", "ecfp")
    assert metadata is not None
    assert metadata.structures_sha256 == second_structures_sha256
    assert metadata.descriptor_identity == second_cfg.identity()


def test_dataset_ids_do_not_share_a_cache_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cfg = EcfpConfig(fingerprint_size=64, name="ecfp")
    for dataset_id in ["lipo", "HIA_Hou"]:
        cfg.load_matrix(
            write_stub_dataset(tmp_path / dataset_id, "a" * 64), dataset_id, cache_dir
        )
    assert sorted(path.name for path in cache_dir.iterdir()) == ["HIA_Hou", "lipo"]


def test_a_dataset_without_provenance_names_the_problem(tmp_path: Path) -> None:
    cfg = EcfpConfig(fingerprint_size=64, name="ecfp")
    ds = stub_dataset(tmp_path / "no_provenance")
    ds.path.mkdir()

    with pytest.raises(FileNotFoundError, match=r"provenance\.yaml"):
        cfg.load_matrix(ds, "toy", tmp_path / "cache")


# ------------------------------------------------ concurrent jobs, broken pairs


class _SlowCountingDescriptor:
    """A cached descriptor whose computation takes a while and is counted."""

    name = "slow"

    def __init__(self) -> None:
        self.computations = 0
        self._counter_lock = threading.Lock()

    def _compute(self) -> np.ndarray:
        with self._counter_lock:
            self.computations += 1
        time.sleep(0.5)
        return np.ones((len(_SMILES), 4), dtype=np.float32)

    def load_matrix(
        self, dataset: MoleculeDataset, dataset_id: str, cache_root: Path
    ) -> np.ndarray:
        return ensure_embeddings(
            cache_root,
            dataset,
            dataset_id=dataset_id,
            descriptor_name=self.name,
            descriptor_identity="slow-v1",
            compute=self._compute,
            producer=producer_record("tests"),
        ).embedding


def test_two_jobs_computing_one_pair_compute_it_once(tmp_path: Path) -> None:
    """Every eval job shares one cache root: the second job waits for the first
    and reuses its pair instead of computing it again over the same files."""
    descriptor = _SlowCountingDescriptor()
    dataset = write_stub_dataset(tmp_path / "toy", "a" * 64)
    cache_dir = tmp_path / "cache"
    results: list[np.ndarray] = []
    errors: list[BaseException] = []

    def job() -> None:
        try:
            results.append(descriptor.load_matrix(dataset, "toy", cache_dir))
        except BaseException as error:  # surfaced below
            errors.append(error)

    threads = [threading.Thread(target=job) for _ in range(2)]
    for thread in threads:
        thread.start()
        time.sleep(0.1)
    for thread in threads:
        thread.join()

    assert errors == []
    assert descriptor.computations == 1
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], results[1])
    assert read_embeddings(cache_dir, "toy", "slow").embedding.shape == (
        len(_SMILES),
        4,
    )
    assert not list((cache_dir / "toy").glob("*partial*"))


def test_a_broken_cached_pair_is_recomputed(tmp_path: Path) -> None:
    cfg = EcfpConfig(fingerprint_size=64, name="ecfp")
    dataset = write_stub_dataset(tmp_path / "toy", "a" * 64)
    cache_dir = tmp_path / "cache"
    expected = cfg.load_matrix(dataset, "toy", cache_dir)
    npz_path, _ = embeddings_paths(cache_dir, "toy", "ecfp")
    np.savez_compressed(
        npz_path,
        structure_id=np.arange(len(_SMILES)),
        embedding=np.zeros((len(_SMILES), 64), dtype=np.float32),
    )

    recomputed = cfg.load_matrix(dataset, "toy", cache_dir)

    np.testing.assert_array_equal(recomputed, expected)
    np.testing.assert_array_equal(
        read_embeddings(cache_dir, "toy", "ecfp").embedding, expected
    )
