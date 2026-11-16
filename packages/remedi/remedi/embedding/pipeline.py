"""Embed every dataset a manifest needs with a Rem3Di checkpoint, then run it.

The yaml holds the checkpoint (``embedding``) and an eval manifest whose
``model`` reads those embeddings back (``kind: precomputed``, same
``name``). Its tasks are benchmark panels and descriptor analyses
(``kind: descriptor_analysis``, :mod:`remedi.latent_evaluation`). One GPU job
runs both halves; ``remedi_eval`` itself never sees the model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from remedi.embedding.calculator import RemediConfig
from remedi.embedding.writer import EmbeddingReport, embed_datasets
from remedi.latent_evaluation.framework_task import DescriptorAnalysisConfig
from remedi_data.yaml_io import write_model
from remedi_eval.descriptors import PrecomputedEmbeddingsConfig
from remedi_eval.framework import BenchmarkPanelConfig, EvalManifest, run_manifest
from remedi_eval.framework.runner import RunReport

logger = logging.getLogger(__name__)

EMBEDDING_REPORT_FILENAME = "embedding_status.yaml"


RemediTaskConfig = Annotated[
    BenchmarkPanelConfig | DescriptorAnalysisConfig,
    Field(discriminator="kind"),
]


class RemediEvalManifest(EvalManifest):
    """An eval manifest whose tasks may also be descriptor analyses."""

    tasks: list[RemediTaskConfig] = Field(default_factory=list, min_length=1)  # type: ignore[assignment]


class EmbedAndEvaluateConfig(BaseModel):
    """A Rem3Di checkpoint plus the manifest that evaluates its embeddings."""

    model_config = ConfigDict(extra="forbid")

    embedding: RemediConfig
    manifest: RemediEvalManifest

    @model_validator(mode="after")
    def _manifest_reads_these_embeddings(self) -> EmbedAndEvaluateConfig:
        model = self.manifest.model
        if not isinstance(model, PrecomputedEmbeddingsConfig):
            raise ValueError(
                "manifest.model must be kind: precomputed, reading the "
                "embeddings this config writes"
            )
        if model.name != self.embedding.name:
            raise ValueError(
                f"manifest.model.name {model.name!r} differs from embedding.name "
                f"{self.embedding.name!r}"
            )
        return self

    @property
    def embeddings_root(self) -> Path:
        model = self.manifest.model
        assert isinstance(model, PrecomputedEmbeddingsConfig)  # checked above
        return model.embeddings_root

    def dataset_dirs(self) -> list[Path]:
        """Every dataset the manifest's tasks read, each once, in task order."""
        seen: dict[Path, None] = {}
        for task in self.manifest.tasks:
            if isinstance(task, BenchmarkPanelConfig):
                for path, _ in task.selected_benchmarks():
                    seen.setdefault(Path(path), None)
            else:
                seen.setdefault(Path(task.dataset_path), None)
        return list(seen)


def embed_and_evaluate(
    config: EmbedAndEvaluateConfig,
) -> tuple[EmbeddingReport, RunReport]:
    """Write the embeddings, record what happened, then run the manifest."""
    output_root = Path(config.manifest.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    embedding_report = embed_datasets(
        config.embedding, config.dataset_dirs(), config.embeddings_root
    )
    write_model(output_root / EMBEDDING_REPORT_FILENAME, embedding_report)
    if embedding_report.failed:
        logger.warning(
            "%d dataset(s) could not be embedded; the panel records them as failures",
            len(embedding_report.failed),
        )
    return embedding_report, run_manifest(config.manifest)
