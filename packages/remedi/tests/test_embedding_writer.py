"""``remedi.embedding``: writing embeddings pairs and the embed-then-evaluate pipeline.

The encoder is replaced by a stub calculator (the GPU path is checked on the
cluster); the checkpoint identity comes from a real ``.pth`` file hash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
import pytest
from pydantic import ValidationError

from remedi.embedding import (
    EmbedAndEvaluateConfig,
    EmbeddingReport,
    RemediConfig,
    embed_and_evaluate,
    embed_datasets,
)
from remedi_data.bundle import EvalMetric, LabelColumn
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.tasks import TaskType
from remedi_data.testing import ACHIRAL_TEN_SMILES, write_small_dataset
from remedi_eval.benchmark.learners import LinearLearnerConfig
from remedi_eval.descriptors import (
    DescriptorCalculator,
    EcfpConfig,
    PrecomputedEmbeddingsConfig,
    read_embeddings,
    read_metadata,
)
from remedi_eval.framework import BenchmarkPanelConfig, EvalManifest

N_ROWS = len(ACHIRAL_TEN_SMILES)


class _EcfpStandIn(DescriptorCalculator):
    """Stands in for the encoder: ECFP-128, counting how often it is called."""

    calls: list[str] = []  # noqa: RUF012 - shared on purpose, reset per test

    def calculate(self, dataset: MoleculeDataset, *, train_indices=None) -> np.ndarray:
        del train_indices
        _EcfpStandIn.calls.append(Path(dataset.path).name)
        return EcfpConfig(name="ecfp_128", length=128).build().calculate(dataset)


@pytest.fixture
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    _EcfpStandIn.calls = []
    monkeypatch.setattr(RemediConfig, "build", lambda self: _EcfpStandIn())
    return _EcfpStandIn.calls


def _checkpoint(tmp_path: Path, weights: bytes = b"weights-v1") -> Path:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "encoder.pth").write_bytes(weights)
    return model_dir


def _datasets(tmp_path: Path, dataset_ids: list[str]) -> list[Path]:
    return [
        write_small_dataset(
            tmp_path / "datasets",
            dataset_id=dataset_id,
            labels=[LabelColumn(name="y", task_type=TaskType.regression)],
            metrics=[EvalMetric.rmse],
            targets=np.linspace(0.0, 1.0, N_ROWS),
        )
        for dataset_id in dataset_ids
    ]


def test_embed_datasets_writes_then_skips_up_to_date_pairs(
    tmp_path: Path, stub_encoder: list[str]
) -> None:
    dataset_dirs = _datasets(tmp_path, ["toy_a", "toy_b"])
    config = RemediConfig(name="remedi_test", model_dir=_checkpoint(tmp_path))

    first = embed_datasets(config, dataset_dirs, tmp_path / "emb")
    second = embed_datasets(config, dataset_dirs, tmp_path / "emb")

    assert first.written == ["toy_a", "toy_b"] and not first.skipped
    assert second.skipped == ["toy_a", "toy_b"] and not second.written
    assert stub_encoder == ["toy_a", "toy_b"]
    metadata = read_metadata(tmp_path / "emb", "toy_a", "remedi_test")
    assert metadata is not None
    assert metadata.descriptor_identity == config.identity()
    assert metadata.producer.package == "remedi"
    embeddings = read_embeddings(tmp_path / "emb", "toy_a", "remedi_test")
    assert embeddings.embedding.shape == (N_ROWS, 128)


def test_new_weights_under_the_same_name_are_re_embedded(
    tmp_path: Path, stub_encoder: list[str]
) -> None:
    dataset_dirs = _datasets(tmp_path, ["toy_a"])
    embed_datasets(
        RemediConfig(name="remedi_test", model_dir=_checkpoint(tmp_path)),
        dataset_dirs,
        tmp_path / "emb",
    )
    retrained = RemediConfig(
        name="remedi_test", model_dir=_checkpoint(tmp_path, b"weights-v2")
    )

    report = embed_datasets(retrained, dataset_dirs, tmp_path / "emb")

    assert report.written == ["toy_a"]
    metadata = read_metadata(tmp_path / "emb", "toy_a", "remedi_test")
    assert metadata is not None and metadata.descriptor_identity == retrained.identity()


def test_one_failing_dataset_is_recorded_and_the_rest_are_embedded(
    tmp_path: Path, stub_encoder: list[str]
) -> None:
    dataset_dirs = _datasets(tmp_path, ["toy_a"])
    broken = tmp_path / "datasets" / "broken"
    broken.mkdir()
    config = RemediConfig(name="remedi_test", model_dir=_checkpoint(tmp_path))

    report = embed_datasets(config, [broken, *dataset_dirs], tmp_path / "emb")

    assert report.written == ["toy_a"]
    assert [failure.dataset_id for failure in report.failed] == ["broken"]


def _pipeline_config(tmp_path: Path, model_name: str = "remedi_test") -> dict:
    return {
        "embedding": {"name": "remedi_test", "model_dir": str(_checkpoint(tmp_path))},
        "manifest": {
            "model": {
                "descriptor_kind": "precomputed",
                "name": model_name,
                "embeddings_root": str(tmp_path / "emb"),
            },
            "output_root": str(tmp_path / "eval_out"),
            "tasks": [
                {
                    "kind": "benchmark_panel",
                    "eval_root": str(tmp_path / "datasets"),
                    "learners": [{"learner_kind": "linear", "ridge_alpha": 1.0}],
                }
            ],
        },
    }


def test_embed_and_evaluate_scores_the_written_embeddings(
    tmp_path: Path, stub_encoder: list[str]
) -> None:
    _datasets(tmp_path, ["toy_a", "toy_b"])
    config = EmbedAndEvaluateConfig.model_validate(_pipeline_config(tmp_path))

    embedding_report, run_report = embed_and_evaluate(config)

    assert embedding_report.written == ["toy_a", "toy_b"]
    assert run_report.n_failed == 0
    results = pd.read_csv(tmp_path / "eval_out" / "benchmark" / "results.csv")
    assert set(results.dataset_id) == {"toy_a", "toy_b"}
    assert set(results.descriptor_name) == {"remedi_test"}
    recorded = pyd_yaml.parse_yaml_file_as(
        EmbeddingReport, tmp_path / "eval_out" / "embedding_status.yaml"
    )
    assert recorded.checkpoint_sha256 == config.embedding.identity()


def test_the_pipeline_refuses_a_manifest_that_reads_other_embeddings(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match=r"differs from embedding\.name"):
        EmbedAndEvaluateConfig.model_validate(_pipeline_config(tmp_path, "other"))
    ecfp_model = _pipeline_config(tmp_path)
    ecfp_model["manifest"]["model"] = {"descriptor_kind": "ecfp", "name": "ecfp"}
    with pytest.raises(ValidationError, match="precomputed"):
        EmbedAndEvaluateConfig.model_validate(ecfp_model)


def test_the_pipeline_embeds_only_the_selected_datasets(tmp_path: Path) -> None:
    _datasets(tmp_path, ["toy_a", "toy_b"])
    raw = _pipeline_config(tmp_path)
    raw["manifest"]["tasks"][0]["dataset_ids"] = ["toy_b"]
    config = EmbedAndEvaluateConfig.model_validate(raw)
    assert [path.name for path in config.dataset_dirs()] == ["toy_b"]
    assert isinstance(config.manifest.model, PrecomputedEmbeddingsConfig)
    assert isinstance(config.manifest.tasks[0], BenchmarkPanelConfig)
    assert isinstance(config.manifest, EvalManifest)
    assert config.manifest.tasks[0].learners == [LinearLearnerConfig(ridge_alpha=1.0)]


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
