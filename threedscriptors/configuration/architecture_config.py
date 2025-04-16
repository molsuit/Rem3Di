import importlib
from collections.abc import Callable, Iterable, Sequence
from enum import Enum

import torch.nn
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from threedscriptors.configuration.config_utils import IrrepType


class HeadType(Enum):
    RESIDUAL = "residual"
    FULLY_CONNECTED = "fully_connected"


class Activations(Enum):
    SILU = "SiLU"
    GELU = "gelu"

    @staticmethod
    def get_activation_fn(activation_name: str) -> Callable:
        if activation_name in Activations.__members__:
            fn_name = Activations.__members__[activation_name].value
            module = importlib.import_module("torch.nn")
            activation_class = getattr(module, fn_name)
            return activation_class()
        else:
            raise ValueError(
                f"Activation function '{activation_name}' is not supported."
            )


class Aggregations(Enum):
    MEAN = "mean"
    MAX = "max"
    STD = "std"

    @staticmethod
    def get_aggregation_fn(aggregation_name: str) -> Callable:
        if aggregation_name in Aggregations.__members__:
            fn_name = Aggregations.__members__[aggregation_name].value
            module = importlib.import_module("torch")
            aggregation_fn = getattr(module, fn_name)
            return aggregation_fn
        else:
            raise ValueError(
                f"Aggregation function '{aggregation_name}' is not supported."
            )


class AttentionLayerConfig(BaseModel):
    num_heads: int
    dim_feedforward: int
    dropout: float
    embedding_dim: int | None = None


class EncoderConfig(BaseModel):
    N_layers: int
    attention_layer_config: AttentionLayerConfig
    reload_state_dict: str | None = None


class RegressionHeadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str | None = None
    activation_fn: Callable = torch.nn.SiLU()
    hidden_dimensions: list[int] = [256, 128]
    input_dimensions: int | None = None
    head_type: HeadType = HeadType.FULLY_CONNECTED

    @field_validator("activation_fn", mode="before")
    @classmethod
    def check_activation_fn(cls, v: str | Callable | Activations) -> Callable:
        if isinstance(v, Activations):
            return Activations.get_activation_fn(v.value.upper())
        elif isinstance(v, Callable):
            return v
        elif isinstance(v, str):
            return Activations.get_activation_fn(v.upper())

    @field_serializer("activation_fn")
    def serialize_activation_fn(self, activation_fn):
        return activation_fn.__class__.__name__


class EmbeddingPreprocessConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pseudoscalars: bool = True
    pseudoscalar_dimension: int  # This does not actually change anything atm, because we have to think more about how to exactly compute the pseudoscalars. Should this be the dimension of the embedding space that gets computed by the linear layers, or should this be the output dimensions of the pseudscalars. It is not clear yet, whether we would actually want to change that, or is only the intermediate spaces should change.
    pseudoscalar_embedding_dim: int | None = None
    input_irreps: IrrepType | None = None
    output_irreps: IrrepType | None = None
    input_embedding_size: int | None = None
    output_irreps_dim: int | None = None
    reload_state_dict: str | None = None


class GlobalAggregatorConfig(BaseModel):
    aggregation_fn: Callable | Iterable[Callable]
    input_dim: int | None = None
    output_dim: int | None = None

    @field_validator("aggregation_fn", mode="before")
    @classmethod
    def check_aggregation_fn(cls, v: str | Callable | Aggregations) -> Callable:
        if isinstance(v, Aggregations):
            return Aggregations.get_aggregation_fn(v.value.upper())
        elif isinstance(v, Callable):
            return v
        elif isinstance(v, str):
            return Aggregations.get_aggregation_fn(v.upper())

    @field_serializer("aggregation_fn")
    def serialize_aggregation_fn(self, aggregation_fn):
        return aggregation_fn.__name__


class ArchitectureConfig(BaseModel):
    embedding_preprocess_config: EmbeddingPreprocessConfig
    encoder_config: EncoderConfig
    global_aggregator_config: GlobalAggregatorConfig
    regression_head_config: RegressionHeadConfig | Sequence[RegressionHeadConfig]
    reload_full_model_weights: str | None = None
