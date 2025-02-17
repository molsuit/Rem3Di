from collections.abc import Callable, Iterable
from dataclasses import dataclass

from e3nn.o3 import Irreps


@dataclass
class AttentionLayerConfig:
    input_dim: int
    embedding_dim: int
    num_heads: int
    dim_feedforward: int
    dropout: float


@dataclass
class ArchitectureConfig:
    N_layers: int
    attention_layer: AttentionLayerConfig


@dataclass
class RegressionHeadConfig:
    activation_fn: Callable
    hidden_dimensions: list[int]


@dataclass
class EmbeddingPreprocessConfig:
    input_irreps: Irreps
    pseudoscalars: bool
    output_irreps: Irreps | None = None
    input_embedding_size: int | None = None
    output_irreps_dim: int | None = None


@dataclass
class GlobalAggregatorConfig:
    aggregation_fn: Callable | Iterable[Callable]
    input_dim: int
    output_dim: int | None = None
