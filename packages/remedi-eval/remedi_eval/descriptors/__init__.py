"""Descriptor calculators the panel evaluates and their yaml configs.

``DescriptorConfig`` is the union a manifest's ``model`` and a panel's
``baseline_descriptors`` accept. Models that ``remedi_eval`` must not import
(Rem3Di and every other trained encoder) enter as precomputed embeddings.
"""

from remedi_eval.descriptors.base import DescriptorCalculator
from remedi_eval.descriptors.cache import compute_and_cache, descriptor_cache_path
from remedi_eval.descriptors.config import DescriptorConfig
from remedi_eval.descriptors.ecfp import EcfpCalculator, EcfpConfig

__all__ = [
    "DescriptorCalculator",
    "DescriptorConfig",
    "EcfpCalculator",
    "EcfpConfig",
    "compute_and_cache",
    "descriptor_cache_path",
]
