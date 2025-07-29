import importlib
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import Literal, Union
import torch.nn
from e3nn.o3 import Irreps
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    field_validator,
    Field
)

from threedscriptors.configuration.config_utils import IrrepType
from threedscriptors.utils.model_utils import (
    get_equivariant_irreps,
    get_invariant_indices,
)

from threedscriptors.model.preprocessing.radial_basis_functions import GaussianBasisFunctions, BesselBasisFunctions
from threedscriptors.configuration.data_config import TaskConfig
from threedscriptors.model.pooling import MeanPool, AttnPool

class HeadType(Enum):
    RESIDUAL = "residual"
    FULLY_CONNECTED = "fully_connected"
    LINEAR = "linear"


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
    d_geo: int| None = None

class DecoderConfig(BaseModel):
    N_layers: int
    d_descriptor: int | None = None
    attention_layer_config: AttentionLayerConfig
    reload_state_dict: str | None = None
    d_pair: int | None = None
    d_geo: int| None = None



class RegressionHeadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str | None = None
    task_config: TaskConfig | None = None
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
    pseudoscalar_dimension: int
    chiral_embedding_dimension: int
    reload_state_dict: str | None = None
    gated: bool = True
    pseudoscalars: bool = True
    equivariant_rms_normalization : bool = True


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

        if self.pseudoscalars:
            odd_invariants_chiral_embedding = Irreps([(self.chiral_embedding_dimension,(0,-1))])
        else:
            odd_invariants_chiral_embedding =  Irreps()

        return even_invariants+odd_invariants_chiral_embedding

    @computed_field(return_type=int, repr = True)
    @property
    def output_irreps_dim(self):
        return self.output_irreps.dim





class Aggregations(Enum):
    MEAN = MeanPool
    ATTENTION = AttnPool

    @classmethod
    def _missing_(cls, value):
        # 1) strings → by name
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                pass

        # 2&3) class or instance → by subclass check
        pool_cls = value if isinstance(value, type) else type(value)
        if issubclass(pool_cls, MeanPool):
            return cls.MEAN
        if issubclass(pool_cls, AttnPool):
            return cls.ATTENTION

        # let Enum blow up otherwise
        return super()._missing_(value)

    def __str__(self) -> str:
        # for repr and JSON serializer
        return self.name.lower()





class MeanAggregatorConfig(BaseModel):
    aggregator_type: Literal[Aggregations.MEAN]


    @field_serializer("aggregator_type")
    def _serialize_aggregator_type(self, v: Aggregations, info):
        return v.name.lower()

    @field_validator("aggregator_type", mode="before")
    @classmethod
    def check_aggregator_type(cls, v: str | Aggregations) -> Callable:
        if isinstance(v, Aggregations):
            return v
        elif isinstance(v, str):
            return Aggregations(v)


class AttentionAggregatorConfig(BaseModel):
    aggregator_type: Literal[Aggregations.ATTENTION]
    num_heads: int
    head_dim: int | None = None
    attn_dropout: float | None = None

    @field_serializer("aggregator_type")
    def _serialize_aggregator_type(self, v: Aggregations, info):
        return v.name.lower()
    

    @field_validator("aggregator_type", mode="before")
    @classmethod
    def check_aggregator_type(cls, v: str | Aggregations) -> Callable:
        if isinstance(v, Aggregations):
            return v
        elif isinstance(v, str):
            return Aggregations(v)


class GlobalAggregatorConfig(BaseModel):
    aggregator_type_config:MeanAggregatorConfig | AttentionAggregatorConfig
    input_dim: int | None = None
    output_dim: int | None = None
    global_molecular_descriptor_dropout : float | None = None

class RandomWalkPositionalEncoding(BaseModel):
    k_hop_random_walk : int
    d_projection: int
    reload_state_dict: str | None = None



class RadialBasisFunctionType(Enum):
    GAUSSIAN = GaussianBasisFunctions
    BESSEL = BesselBasisFunctions

    @classmethod
    def _missing_(cls, value):
        # 1) strings → by name
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                print("broken key")
                raise




class RelativeDistancePositionalEncodingConfig(BaseModel):
    N_radial_basis_functions: int
    distance_cutoff: float
    d_projection: int
    basis_function_type: RadialBasisFunctionType = RadialBasisFunctionType.GAUSSIAN
    reload_state_dict: str| None = None

    @field_serializer("basis_function_type")
    def _serialize_aggregator_type(self, v: RadialBasisFunctionType, info):
        return v.name.lower()
    

    @field_validator("basis_function_type", mode="before")
    @classmethod
    def check_aggregator_type(cls, v: str | RadialBasisFunctionType) -> Callable:
        if isinstance(v, RadialBasisFunctionType):
            return v
        elif isinstance(v, str):
            return RadialBasisFunctionType(v)




class ArchitectureConfig(BaseModel):
    embedding_preprocess_config: EmbeddingPreprocessConfig
    encoder_config: EncoderConfig
    global_aggregator_config: GlobalAggregatorConfig
    regression_head_config: RegressionHeadConfig | Sequence[RegressionHeadConfig] | None
    positional_encoding_config: RelativeDistancePositionalEncodingConfig | RandomWalkPositionalEncoding | None = None
    reload_full_model_weights: str | None = None
    decoder_config: DecoderConfig | None = None



