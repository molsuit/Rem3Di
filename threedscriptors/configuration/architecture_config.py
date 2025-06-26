import importlib
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import Optional
import torch.nn
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, computed_field
from e3nn.o3 import Irreps
from threedscriptors.utils.model_utils import  get_equivariant_irreps, get_invariant_indices
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
    d_pair: int | None = None


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

    input_irreps: IrrepType | None = None
    pseudoscalars: bool = True
    pseudoscalar_dimension: int  
    
    reload_state_dict: str | None = None

    pseudoscalar_embedding_dim: int | None = None


    @computed_field(return_type=IrrepType, repr= True)
    @property
    def pseudoscalar_irrep(self):

        if self.pseudoscalars:
            return Irreps([(self.pseudoscalar_dimension,(0,-1))])
        else:
            return Irreps()


    @computed_field(return_type=int, repr=True)
    @property
    def input_equivariant_dimension(self) -> int | None: 
        equivariant_irreps = get_equivariant_irreps(self.input_irreps)
        return sum([e_irrep.dim for e_irrep in equivariant_irreps])
    
    @computed_field(return_type=int, repr=True)
    @property
    def input_dimension(self):
        return self.input_irreps.dim
    
    @computed_field(return_type=int, repr=True)
    @property
    def input_invariant_dimension(self):
        invariant_irreps = [Irreps([(m, (i.l, i.p))])  for m, i in self.input_irreps if i.l == 0]

        return sum([i_irrep.dim for i_irrep in invariant_irreps])
    

    @computed_field(return_type= IrrepType, repr= True)
    @property
    def output_irreps(self):

        _, even_invariants = get_invariant_indices(self.input_irreps)

        odd_invariants = self.pseudoscalar_irrep


        return even_invariants+odd_invariants

    @computed_field(return_type=int, repr = True)
    @property
    def output_irreps_dim(self):
        return self.output_irreps.dim


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



class PositionalEncodingConfig(BaseModel):
    N_radial_basis_functions: int
    distance_cutoff: float
    d_projection: int


class ArchitectureConfig(BaseModel):
    embedding_preprocess_config: EmbeddingPreprocessConfig
    encoder_config: EncoderConfig
    global_aggregator_config: GlobalAggregatorConfig
    regression_head_config: RegressionHeadConfig | Sequence[RegressionHeadConfig]
    positional_encoding_config: Optional[PositionalEncodingConfig] = None
    reload_full_model_weights: str | None = None



