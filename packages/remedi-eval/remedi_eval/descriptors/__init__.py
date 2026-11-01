"""Descriptor calculators the panel evaluates and their yaml configs.

``DescriptorConfig`` is the union a manifest's ``model`` and a panel's
``baseline_descriptors`` accept: ECFP, computed and cached here, and
precomputed embeddings, the entry for every trained encoder.
"""

from remedi_eval.descriptors.base import DescriptorCalculator
from remedi_eval.descriptors.cache import compute_and_cache
from remedi_eval.descriptors.config import DescriptorConfig
from remedi_eval.descriptors.ecfp import EcfpCalculator, EcfpConfig
from remedi_eval.descriptors.embeddings_file import (
    EmbeddingsFile,
    EmbeddingsFileError,
    EmbeddingsMetadata,
    ProducerRecord,
    embeddings_aligned_to,
    embeddings_paths,
    producer_record,
    read_embeddings,
    read_metadata,
    write_embeddings,
)
from remedi_eval.descriptors.precomputed import PrecomputedEmbeddingsConfig

__all__ = [
    "DescriptorCalculator",
    "DescriptorConfig",
    "EcfpCalculator",
    "EcfpConfig",
    "EmbeddingsFile",
    "EmbeddingsFileError",
    "EmbeddingsMetadata",
    "PrecomputedEmbeddingsConfig",
    "ProducerRecord",
    "compute_and_cache",
    "embeddings_aligned_to",
    "embeddings_paths",
    "producer_record",
    "read_embeddings",
    "read_metadata",
    "write_embeddings",
]
