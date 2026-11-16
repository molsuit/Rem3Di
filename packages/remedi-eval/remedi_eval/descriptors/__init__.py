"""Descriptor calculators the panel evaluates and their yaml configs.

``DescriptorConfig`` is the union a manifest's ``model`` and a panel's
``baseline_descriptors`` accept: ECFP, computed and cached here, and
precomputed embeddings, the entry for every trained encoder.
"""

from remedi_eval.descriptors.config import DescriptorConfig
from remedi_eval.descriptors.ecfp import EcfpConfig
from remedi_eval.descriptors.embeddings_file import (
    EmbeddingsFile,
    EmbeddingsFileError,
    EmbeddingsMetadata,
    EnsuredEmbeddings,
    ProducerRecord,
    embeddings_aligned_to,
    embeddings_paths,
    ensure_embeddings,
    producer_record,
    read_embeddings,
    read_metadata,
    write_embeddings,
)
from remedi_eval.descriptors.precomputed import PrecomputedEmbeddingsConfig

__all__ = [
    "DescriptorConfig",
    "EcfpConfig",
    "EmbeddingsFile",
    "EmbeddingsFileError",
    "EmbeddingsMetadata",
    "EnsuredEmbeddings",
    "PrecomputedEmbeddingsConfig",
    "ProducerRecord",
    "embeddings_aligned_to",
    "embeddings_paths",
    "ensure_embeddings",
    "producer_record",
    "read_embeddings",
    "read_metadata",
    "write_embeddings",
]
