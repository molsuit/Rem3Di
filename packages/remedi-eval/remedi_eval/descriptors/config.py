"""The descriptor-config union a manifest's ``model`` and a panel's baselines accept."""

from typing import Annotated

from pydantic import Field

from remedi_eval.descriptors.ecfp import EcfpConfig
from remedi_eval.descriptors.precomputed import PrecomputedEmbeddingsConfig

DescriptorConfig = Annotated[
    EcfpConfig | PrecomputedEmbeddingsConfig, Field(discriminator="descriptor_kind")
]
