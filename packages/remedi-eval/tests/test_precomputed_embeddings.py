"""Precomputed embeddings: the npz + sidecar yaml pair, and the descriptor that reads it.

The pair is the entry point for every frozen model (``BENCHMARK_DATA_FORMAT.md``
§7.5). These tests pin what the reader refuses, that it reorders to the zarr's
structure order, and that a panel scored on precomputed embeddings gives the
same numbers as the same matrix computed in-process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from remedi_data.bundle import EvalMetric, LabelColumn, structures_identity
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.tasks import TaskType
from remedi_data.testing import ACHIRAL_TEN_SMILES, write_small_dataset
from remedi_eval.benchmark.learners import LinearLearnerConfig
from remedi_eval.descriptors import (
    EcfpConfig,
    EmbeddingsFileError,
    PrecomputedEmbeddingsConfig,
    ProducerRecord,
    embeddings_aligned_to,
    embeddings_paths,
    read_embeddings,
    write_embeddings,
)
from remedi_eval.framework import BenchmarkPanelConfig, EvalManifest, run_manifest

N_ROWS = len(ACHIRAL_TEN_SMILES)
PRODUCER = ProducerRecord(package="tests", version="0")


def _dataset(root: Path, dataset_id: str = "toy") -> Path:
    return write_small_dataset(
        root / "datasets",
        dataset_id=dataset_id,
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.rmse],
        targets=np.linspace(0.0, 1.0, N_ROWS),
    )


def _write(
    embeddings_root: Path,
    dataset_dir: Path,
    structure_id: np.ndarray,
    embedding: np.ndarray,
    *,
    dataset_id: str = "toy",
    structures_sha256: str | None = None,
) -> None:
    write_embeddings(
        embeddings_root,
        dataset_id=dataset_id,
        descriptor_name="frozen",
        descriptor_identity="checkpoint-sha",
        structures_sha256=structures_sha256 or structures_identity(dataset_dir),
        structure_id=structure_id,
        embedding=embedding,
        producer=PRODUCER,
    )


def _structure_ids(dataset_dir: Path) -> np.ndarray:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
    return np.asarray(dataset.structure_ids[:], dtype=np.int64)


def test_a_shuffled_file_is_returned_in_zarr_order(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    ids = _structure_ids(dataset_dir)
    matrix = np.arange(N_ROWS * 3, dtype=np.float32).reshape(N_ROWS, 3)
    order = np.random.default_rng(0).permutation(N_ROWS)
    _write(tmp_path / "emb", dataset_dir, ids[order], matrix[order])

    embeddings = read_embeddings(tmp_path / "emb", "toy", "frozen")
    aligned = embeddings_aligned_to(embeddings, dataset_dir, ids)

    np.testing.assert_array_equal(aligned, matrix)
    assert embeddings.metadata.n_structures == N_ROWS
    assert embeddings.metadata.embedding_dimension == 3


def test_embeddings_for_another_build_are_refused(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    ids = _structure_ids(dataset_dir)
    _write(
        tmp_path / "emb",
        dataset_dir,
        ids,
        np.zeros((N_ROWS, 2)),
        structures_sha256="0" * 64,
    )
    embeddings = read_embeddings(tmp_path / "emb", "toy", "frozen")
    with pytest.raises(EmbeddingsFileError, match="recompute them"):
        embeddings_aligned_to(embeddings, dataset_dir, ids)


@pytest.mark.parametrize(
    "structure_ids",
    [np.arange(N_ROWS - 1), np.arange(1, N_ROWS + 1)],
    ids=["missing_structure", "foreign_structure"],
)
def test_embeddings_covering_other_structures_are_refused(
    tmp_path: Path, structure_ids: np.ndarray
) -> None:
    dataset_dir = _dataset(tmp_path)
    _write(
        tmp_path / "emb",
        dataset_dir,
        structure_ids,
        np.zeros((len(structure_ids), 2)),
    )
    embeddings = read_embeddings(tmp_path / "emb", "toy", "frozen")
    with pytest.raises(EmbeddingsFileError, match="cover"):
        embeddings_aligned_to(embeddings, dataset_dir, _structure_ids(dataset_dir))


def test_an_npz_that_does_not_match_its_yaml_is_refused(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    ids = _structure_ids(dataset_dir)
    _write(tmp_path / "emb", dataset_dir, ids, np.zeros((N_ROWS, 2)))
    npz_path, _ = embeddings_paths(tmp_path / "emb", "toy", "frozen")
    np.savez_compressed(npz_path, structure_id=ids, embedding=np.ones((N_ROWS, 2)))

    with pytest.raises(EmbeddingsFileError, match="npz_sha256"):
        read_embeddings(tmp_path / "emb", "toy", "frozen")


def test_an_npz_without_its_yaml_is_not_a_pair(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    _write(
        tmp_path / "emb",
        dataset_dir,
        _structure_ids(dataset_dir),
        np.zeros((N_ROWS, 2)),
    )
    _, yaml_path = embeddings_paths(tmp_path / "emb", "toy", "frozen")
    yaml_path.unlink()

    with pytest.raises(FileNotFoundError, match=r"frozen\.yaml"):
        read_embeddings(tmp_path / "emb", "toy", "frozen")


@pytest.mark.parametrize(
    ("structure_id", "embedding", "message"),
    [
        (np.array([0, 0, 1]), np.zeros((3, 2)), "not unique"),
        (np.arange(3), np.zeros((2, 2)), "embedding rows"),
        (np.arange(2), np.array([[0.0, np.nan], [1.0, 1.0]]), "NaN or inf"),
        (np.arange(2), np.zeros(2), "2-D"),
    ],
    ids=["repeated_ids", "row_count", "non_finite", "one_dimensional"],
)
def test_the_writer_refuses_malformed_arrays(
    tmp_path: Path, structure_id: np.ndarray, embedding: np.ndarray, message: str
) -> None:
    with pytest.raises(EmbeddingsFileError, match=message):
        write_embeddings(
            tmp_path / "emb",
            dataset_id="toy",
            descriptor_name="frozen",
            descriptor_identity="x",
            structures_sha256="a" * 64,
            structure_id=structure_id,
            embedding=embedding,
            producer=PRODUCER,
        )
    assert not embeddings_paths(tmp_path / "emb", "toy", "frozen")[1].exists()


def _panel(
    tmp_path: Path, model: EcfpConfig | PrecomputedEmbeddingsConfig, name: str
) -> pd.DataFrame:
    output_root = tmp_path / "eval_out" / name
    run_manifest(
        EvalManifest(
            model=model,
            output_root=output_root,
            resource_cache_dir=tmp_path / "cache",
            tasks=[
                BenchmarkPanelConfig(
                    eval_root=tmp_path / "datasets",
                    learners=[LinearLearnerConfig(ridge_alpha=1.0)],
                )
            ],
        )
    )
    return pd.read_csv(output_root / "benchmark" / "results.csv")


def test_a_panel_on_precomputed_embeddings_matches_the_in_process_descriptor(
    tmp_path: Path,
) -> None:
    dataset_dir = _dataset(tmp_path)
    ecfp = EcfpConfig(name="ecfp_128", length=128)
    in_process = _panel(tmp_path, ecfp, "in_process")

    # The same matrix, written by "another package" in shuffled order.
    dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
    matrix = ecfp.build().calculate(dataset)
    ids = _structure_ids(dataset_dir)
    order = np.random.default_rng(1).permutation(N_ROWS)
    _write(tmp_path / "emb", dataset_dir, ids[order], matrix[order])
    precomputed = _panel(
        tmp_path,
        PrecomputedEmbeddingsConfig(name="frozen", embeddings_root=tmp_path / "emb"),
        "precomputed",
    )

    columns = ["dataset_id", "learner_kind", "metric_name", "metric_value", "n_test"]
    pd.testing.assert_frame_equal(in_process[columns], precomputed[columns])
    assert set(precomputed.descriptor_name) == {"frozen"}


def test_a_panel_without_embeddings_for_a_dataset_records_the_failure(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path)
    output_root = tmp_path / "eval_out" / "missing"
    run_manifest(
        EvalManifest(
            model=PrecomputedEmbeddingsConfig(
                name="frozen", embeddings_root=tmp_path / "emb"
            ),
            output_root=output_root,
            tasks=[
                BenchmarkPanelConfig(
                    eval_root=tmp_path / "datasets",
                    learners=[LinearLearnerConfig(ridge_alpha=1.0)],
                )
            ],
        )
    )
    failures = (output_root / "benchmark" / "failures.yaml").read_text()
    assert "toy" in failures and "frozen.yaml" in failures
