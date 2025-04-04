from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import importlib

from e3nn.o3 import Irreps

class Activations(Enum):
    SILU = "silu"
    GELU = "gelu"

    @staticmethod
    def get_activation_fn(activation_name: str) -> Callable:
        if activation_name in Activations.__members__:
            module = importlib.import_module("torch.nn")
            activation_class = getattr(module, activation_name)
            return activation_class()
        else:
            raise ValueError(f"Activation function '{activation_name}' is not supported.")
        


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
    name:str
    activation_fn: Activations = Activations.SILU
    hidden_dimensions: list[int]


@dataclass
class EmbeddingPreprocessConfig:
    input_irreps: Irreps
    pseudoscalars: bool
    output_irreps: Irreps | None = None
    input_embedding_size: int | None = None
    output_irreps_dim: int | None = None

    def serialize(self):
        return {
            "input_irreps": str(self.input_irreps),
            "pseudoscalars": self.pseudoscalars,
            "output_irreps": str(self.output_irreps),
            "input_embedding_size": self.input_embedding_size,
            "output_irreps_dim": self.output_irreps_dim,
        }


@dataclass
class GlobalAggregatorConfig:
    aggregation_fn: Callable | Iterable[Callable]
    input_dim: int
    output_dim: int | None = None
