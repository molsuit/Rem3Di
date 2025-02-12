from collections.abc import Callable
from dataclasses import dataclass


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
