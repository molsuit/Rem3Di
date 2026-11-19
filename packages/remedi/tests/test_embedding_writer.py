"""``remedi.embedding``: writing embeddings pairs and the embed-then-evaluate pipeline.

The encoder is replaced by a stub calculator (the GPU path is checked on the
cluster); the checkpoint identity comes from a real ``.pth`` file hash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from remedi.embedding import (
    EmbedAndEvaluateConfig,
    EmbeddingReport,
    MaceMeanConfig,
    RemediConfig,
    embed_and_evaluate,
    embed_datasets,
)
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, ENCODER_WEIGHT_FILES
from remedi_data.dataset import Dataset
from remedi_data.format import EvalMetric, LabelColumn, TaskType
from remedi_data.yaml_io import read_model
from remedi_eval.benchmark.learners import LinearLearner
from remedi_eval.descriptors import (
    EcfpConfig,
    PrecomputedEmbeddingsConfig,
    read_embeddings,
    read_metadata,
)
from remedi_eval.framework import BenchmarkPanelConfig, EvalManifest
from testing import ACHIRAL_TEN_SMILES, write_small_dataset

N_ROWS = len(ACHIRAL_TEN_SMILES)


class _EcfpStandIn:
    """Stands in for the encoder: ECFP-128, counting how often it is called."""

    calls: list[str] = []  # noqa: RUF012 - shared on purpose, reset per test

    def calculate(self, dataset: Dataset) -> np.ndarray:
        _EcfpStandIn.calls.append(Path(dataset.path).name)
        return EcfpConfig(name="ecfp_128", fingerprint_size=128).calculate(dataset)


@pytest.fixture
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    _EcfpStandIn.calls = []
    monkeypatch.setattr(RemediConfig, "build", lambda self: _EcfpStandIn())
    return _EcfpStandIn.calls


def _checkpoint(
    tmp_path: Path,
    weights: bytes = b"weights-v1",
    architecture: str = "mace_config: {model_path: null}\n",
    name: str = "checkpoint",
) -> Path:
    """A checkpoint directory with stand-in files; only their bytes are hashed."""
    model_dir = tmp_path / name
    model_dir.mkdir(exist_ok=True)
    (model_dir / ARCHITECTURE_FILENAME).write_text(architecture)
    for filename in ENCODER_WEIGHT_FILES:
        (model_dir / filename).write_bytes(weights + filename.encode())
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
        "embedding": {
            "kind": "remedi",
            "name": "remedi_test",
            "model_dir": str(_checkpoint(tmp_path)),
        },
        "manifest": {
            "model": {
                "kind": "precomputed",
                "name": model_name,
                "embeddings_root": str(tmp_path / "emb"),
            },
            "output_root": str(tmp_path / "eval_out"),
            "tasks": [
                {
                    "kind": "benchmark_panel",
                    "eval_root": str(tmp_path / "datasets"),
                    "learners": [{"kind": "linear", "ridge_alpha": 1.0}],
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
    recorded = read_model(
        tmp_path / "eval_out" / "embedding_status.yaml", EmbeddingReport
    )
    assert recorded.checkpoint_sha256 == config.embedding.identity()


def test_the_pipeline_refuses_a_manifest_that_reads_other_embeddings(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match=r"differs from embedding\.name"):
        EmbedAndEvaluateConfig.model_validate(_pipeline_config(tmp_path, "other"))
    ecfp_model = _pipeline_config(tmp_path)
    ecfp_model["manifest"]["model"] = {"kind": "ecfp", "name": "ecfp"}
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
    assert config.manifest.tasks[0].learners == [LinearLearner(ridge_alpha=1.0)]


def test_the_remedi_identity_is_the_checkpoint_hash_and_is_memoised(
    tmp_path: Path,
) -> None:
    model_dir = _checkpoint(tmp_path)
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
    other_dir = _checkpoint(tmp_path, b"weights-v2", name="other")
    assert RemediConfig(name="remedi", model_dir=other_dir).identity() != identity

    # A model directory without one of its files names the problem.
    (other_dir / ENCODER_WEIGHT_FILES[1]).unlink()
    with pytest.raises(FileNotFoundError, match=ENCODER_WEIGHT_FILES[1]):
        RemediConfig(name="remedi", model_dir=other_dir).identity()


def test_the_remedi_identity_covers_the_architecture_and_the_mace_model(
    tmp_path: Path,
) -> None:
    """The same weights under another architecture yaml, or with another MACE
    model file, compute other descriptors and must not share embeddings."""
    baseline = RemediConfig(name="remedi", model_dir=_checkpoint(tmp_path)).identity()

    other_architecture = _checkpoint(
        tmp_path,
        architecture="mace_config: {model_path: null}\nn_layers: 3\n",
        name="other_architecture",
    )
    assert RemediConfig(name="remedi", model_dir=other_architecture).identity() != (
        baseline
    )

    first_mace, second_mace = tmp_path / "first.model", tmp_path / "second.model"
    first_mace.write_bytes(b"mace-a")
    second_mace.write_bytes(b"mace-b")
    with_first = RemediConfig(
        name="remedi", model_dir=_checkpoint(tmp_path), mace_model_path=first_mace
    ).identity()
    with_second = RemediConfig(
        name="remedi", model_dir=_checkpoint(tmp_path), mace_model_path=second_mace
    ).identity()
    assert len({baseline, with_first, with_second}) == 3

    # The MACE file the yaml names counts too, and a missing one is an error.
    named_in_yaml = _checkpoint(
        tmp_path,
        architecture=f"mace_config: {{model_path: {first_mace}}}\n",
        name="named_in_yaml",
    )
    assert RemediConfig(name="remedi", model_dir=named_in_yaml).identity() != baseline
    first_mace.unlink()
    with pytest.raises(FileNotFoundError, match=r"first\.model"):
        RemediConfig(name="remedi", model_dir=named_in_yaml).identity()


def test_embed_and_evaluate_runs_descriptor_analyses_on_the_embeddings(
    tmp_path: Path, stub_encoder: list[str]
) -> None:
    """A ``descriptor_analysis`` task gets its dataset embedded and analysed in one run."""
    (corpus,) = _datasets(tmp_path, ["corpus"])
    raw = _pipeline_config(tmp_path)
    raw["manifest"]["tasks"] = [
        {
            "kind": "descriptor_analysis",
            "dataset_path": str(corpus),
            "tasks": [{"kind": "capacity_diagnostic"}],
        }
    ]
    config = EmbedAndEvaluateConfig.model_validate(raw)
    assert config.dataset_dirs() == [corpus]

    embedding_report, run_report = embed_and_evaluate(config)

    assert embedding_report.written == ["corpus"]
    assert stub_encoder == ["corpus"]
    assert run_report.n_failed == 0
    output = tmp_path / "eval_out" / "descriptor_analysis" / "capacity_diagnostic.yaml"
    assert output.exists()


def test_the_mace_mean_identity_ignores_the_encoder_weights(tmp_path: Path) -> None:
    """The mean-pooled MACE baseline depends on the MACE model and the layer
    selection only: retraining the encoder must not force a re-embedding, and
    it must never share embeddings with the Rem3Di descriptor."""
    architecture = (
        "mace_config: {model_path: null}\n"
        "embedding_preprocess_config: {mace_layer_indices: [1]}\n"
    )
    first = _checkpoint(tmp_path, b"weights-v1", architecture=architecture)
    retrained = _checkpoint(
        tmp_path, b"weights-v2", architecture=architecture, name="retrained"
    )
    other_layer = _checkpoint(
        tmp_path,
        architecture=architecture.replace("[1]", "[0]"),
        name="other_layer",
    )
    identity = MaceMeanConfig(name="mace_mean", model_dir=first).identity()
    assert MaceMeanConfig(name="mace_mean", model_dir=retrained).identity() == identity
    assert MaceMeanConfig(name="mace_mean", model_dir=other_layer).identity() != (
        identity
    )
    assert RemediConfig(name="mace_mean", model_dir=first).identity() != identity


def test_the_embedding_kind_selects_the_descriptor(tmp_path: Path) -> None:
    model_dir = _checkpoint(tmp_path)
    manifest = _pipeline_config(tmp_path, model_name="mace_mean")["manifest"]
    config = EmbedAndEvaluateConfig.model_validate(
        {
            "embedding": {
                "kind": "mace_mean",
                "name": "mace_mean",
                "model_dir": str(model_dir),
            },
            "manifest": manifest,
        }
    )
    assert isinstance(config.embedding, MaceMeanConfig)
    with pytest.raises(ValidationError):
        EmbedAndEvaluateConfig.model_validate(
            {
                "embedding": {
                    "kind": "average",
                    "name": "mace_mean",
                    "model_dir": str(model_dir),
                },
                "manifest": manifest,
            }
        )
