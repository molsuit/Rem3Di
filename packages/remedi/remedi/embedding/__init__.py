"""Embedding datasets with a trained Rem3Di encoder, as precomputed-embeddings files."""

from remedi.embedding.calculator import RemediCalculator, RemediConfig
from remedi.embedding.pipeline import EmbedAndEvaluateConfig, embed_and_evaluate
from remedi.embedding.writer import EmbeddingReport, embed_datasets

__all__ = [
    "EmbedAndEvaluateConfig",
    "EmbeddingReport",
    "RemediCalculator",
    "RemediConfig",
    "embed_and_evaluate",
    "embed_datasets",
]
