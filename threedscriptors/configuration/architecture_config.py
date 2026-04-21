import importlib
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from threedscriptors.configuration.config_utils import IrrepType
from threedscriptors.model.pooling import AttnPool, MeanPool, PMAAggregator
from threedscriptors.model.preprocessing.radial_basis_functions import (
    BesselBasisFunctions,
    GaussianBasisFunctions,
)
from threedscriptors.utils.model_utils import (
    get_equivariant_irreps,
    get_invariant_indices,
)

if TYPE_CHECKING:
    from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
        AtomicDescriptorPreprocessor,
    )
    from threedscriptors.model.preprocessing.geometric_preprocessor import (
        PairDistanceMatrixGeometricPreprocessor,
    )


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
    d_geo: int | None = None

    def build(self, global_aggregator: nn.Module) -> nn.Module:
        from threedscriptors.model.pair_encoder import TransformerPairEncoder

        encoder = TransformerPairEncoder(self, global_aggregator)
        if self.reload_state_dict:
            encoder.load_state_dict(torch.load(self.reload_state_dict), strict=False)
        return encoder


class DecoderConfig(BaseModel):
    N_layers: int
    d_descriptor: int | None = None
    attention_layer_config: AttentionLayerConfig
    reload_state_dict: str | None = None
    d_pair: int | None = None
    d_geo: int | None = None

    def build(self) -> nn.Module:
        from threedscriptors.model.decoder import TransformerPairDecoder

        decoder = TransformerPairDecoder(self)
        if self.reload_state_dict:
            decoder.load_state_dict(torch.load(self.reload_state_dict))
        return decoder


class RegressionHeadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str | None = None
    task_config: None = None
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


class PrecomputedInvariantNormalizationConfig(BaseModel):
    kind: Literal["precomputed_normalization"] = "precomputed_normalization"

    def build(self, invariant_dimension: int) -> nn.Module:
        from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
            PrecomputedInvariantNormalization,
        )

        return PrecomputedInvariantNormalization(
            invariant_dimension=invariant_dimension
        )


class OnTheFlyInvariantNormalizationConfig(BaseModel):
    kind: Literal["on_the_fly_normalization"] = "on_the_fly_normalization"
    momentum: float
    warm_up_batches: int

    def build(self, invariant_dimension: int) -> nn.Module:
        from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
            OnTheFlyInvariantNormalization,
        )

        return OnTheFlyInvariantNormalization(
            invariant_dimension=invariant_dimension,
            momentum=self.momentum,
            warmup_batches=self.warm_up_batches,
        )


InvNormConfig = Annotated[
    PrecomputedInvariantNormalizationConfig | OnTheFlyInvariantNormalizationConfig,
    Field(discriminator="kind"),
]


class EmbeddingPreprocessConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_irreps: IrrepType
    pseudoscalar_dimension: int
    chiral_embedding_dimension: int
    reload_state_dict: str | None = None
    gated: bool = True
    pseudoscalars: bool = True
    equivariant_rms_normalization: bool = True
    invariant_normalization_config: InvNormConfig = (
        PrecomputedInvariantNormalizationConfig()
    )

    @computed_field(return_type=IrrepType, repr=True)
    @property
    def pseudoscalar_irrep(self):
        if self.pseudoscalars:
            return Irreps([(self.pseudoscalar_dimension, (0, -1))])
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
        invariant_irreps = [
            Irreps([(m, (i.l, i.p))]) for m, i in self.input_irreps if i.l == 0
        ]

        return sum([i_irrep.dim for i_irrep in invariant_irreps])

    @computed_field(return_type=IrrepType, repr=True)
    @property
    def output_irreps(self):
        _, even_invariants = get_invariant_indices(self.input_irreps)

        if self.pseudoscalars:
            odd_invariants_chiral_embedding = Irreps(
                [(self.chiral_embedding_dimension, (0, -1))]
            )
        else:
            odd_invariants_chiral_embedding = Irreps()

        return even_invariants + odd_invariants_chiral_embedding

    @computed_field(return_type=int, repr=True)
    @property
    def output_irreps_dim(self):
        return self.output_irreps.dim

    @computed_field(return_type=IrrepType, repr=True)
    @property
    def invariant_irreps(self):
        _, irreps = get_invariant_indices(self.input_irreps)
        return irreps

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "AtomicDescriptorPreprocessor":
        from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
            AtomicDescriptorPreprocessor,
            PrecomputedInvariantNormalization,
        )

        invariant_normalization = self.invariant_normalization_config.build(
            invariant_dimension=self.invariant_irreps.dim
        )

        if (
            isinstance(invariant_normalization, PrecomputedInvariantNormalization)
            and mean_atomic_embedding is not None
            and std_atomic_embedding is not None
        ):
            assert mean_atomic_embedding.shape[-1] == self.invariant_irreps.dim
            invariant_normalization.set_stats(
                mean=mean_atomic_embedding, std=std_atomic_embedding
            )

        atomic_preprocessor = AtomicDescriptorPreprocessor(
            preprocess_config=self,
            invariant_normalization=invariant_normalization,
        )

        if self.reload_state_dict is not None:
            atomic_preprocessor.load_state_dict(torch.load(self.reload_state_dict))

        return atomic_preprocessor


class Aggregations(Enum):
    MEAN = MeanPool
    ATTENTION = AttnPool
    PMA_ATTENTION = PMAAggregator

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
        if issubclass(pool_cls, PMAAggregator):
            return cls.PMA_ATTENTION

        return super()._missing_(value)

    def __str__(self) -> str:
        return self.name.lower()


def _to_discriminator(value) -> str:
    """Normalize any accepted input to the lowercase discriminator string."""
    if isinstance(value, str):
        return value.strip().lower()
    agg = Aggregations(value)
    return agg.name.lower()


class MeanAggregatorConfig(BaseModel):
    aggregator_type: Literal["mean"] = "mean"

    def build(self, input_dim: int, output_dim: int) -> nn.Module:
        return MeanPool()


class AttentionAggregatorConfig(BaseModel):
    aggregator_type: Literal["attention"] = "attention"
    num_heads: int
    head_dim: int | None = None
    attn_dropout: float | None = None

    def build(self, input_dim: int, output_dim: int) -> nn.Module:
        return AttnPool(
            d_in=input_dim,
            d_hidden=self.head_dim,
            n_heads=self.num_heads,
            dropout=self.attn_dropout or 0.0,
        )


class PMAAggregatorConfig(BaseModel):
    aggregator_type: Literal["pma_attention"] = "pma_attention"
    head_dim: int
    num_heads: int = 4
    attn_dropout: float = 0.0
    num_seeds: int = 16
    reduction: Literal["mean", "sum", "max"] = "mean"
    use_mlp: bool = False

    def build(self, input_dim: int, output_dim: int) -> nn.Module:
        return PMAAggregator(
            d_in=input_dim,
            d_out=output_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            k_seeds=self.num_seeds,
            dropout=self.attn_dropout,
            use_mlp=self.use_mlp,
        )


AggUnion = Annotated[
    MeanAggregatorConfig | AttentionAggregatorConfig | PMAAggregatorConfig,
    Field(discriminator="aggregator_type"),
]


class GlobalAggregatorConfig(BaseModel):
    aggregator_type_config: AggUnion
    input_dim: int | None = None
    output_dim: int | None = None
    global_molecular_descriptor_dropout: float | None = None

    # Normalize BEFORE discriminated-union selection happens
    @model_validator(mode="before")
    @classmethod
    def _normalize_discriminator(cls, data):
        if isinstance(data, dict) and "aggregator_type_config" in data:
            cfg = data["aggregator_type_config"]
            if isinstance(cfg, dict) and "aggregator_type" in cfg:
                try:
                    cfg["aggregator_type"] = _to_discriminator(cfg["aggregator_type"])
                except Exception:
                    pass
                data["aggregator_type_config"] = cfg
        return data

    def build(self) -> nn.Module:
        from threedscriptors.model.global_aggregator import GlobalAggregator

        return GlobalAggregator(self)


class RadialBasisFunctionType(Enum):
    GAUSSIAN = GaussianBasisFunctions
    BESSEL = BesselBasisFunctions

    @classmethod
    def _missing_(cls, value):
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
    reload_state_dict: str | None = None

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

    def build(self) -> "PairDistanceMatrixGeometricPreprocessor":
        from threedscriptors.model.preprocessing.geometric_preprocessor import (
            PairDistanceMatrixGeometricPreprocessor,
        )

        module = PairDistanceMatrixGeometricPreprocessor(
            N_radial_basis_functions=self.N_radial_basis_functions,
            distance_cutoff=self.distance_cutoff,
            d_projection=self.d_projection,
            basis_function_type=self.basis_function_type,
        )
        if self.reload_state_dict is not None:
            module.load_state_dict(torch.load(self.reload_state_dict))
        return module


class ArchitectureConfig(BaseModel):
    embedding_preprocess_config: EmbeddingPreprocessConfig
    encoder_config: EncoderConfig
    global_aggregator_config: GlobalAggregatorConfig
    regression_head_config: RegressionHeadConfig | Sequence[RegressionHeadConfig] | None
    positional_encoding_config: RelativeDistancePositionalEncodingConfig
    reload_full_model_weights: str | None = None
    decoder_config: DecoderConfig | None = None

    @model_validator(mode="after")
    def _cascade_derived_fields(self):
        embed_dim = self.embedding_preprocess_config.output_irreps_dim
        pos = self.positional_encoding_config

        _fill_encoder_dims(self.encoder_config, embed_dim, pos)
        _fill_aggregator_dims(
            self.global_aggregator_config,
            self.encoder_config.attention_layer_config.embedding_dim,
        )
        if self.decoder_config is not None:
            _fill_decoder_dims(
                self.decoder_config,
                embed_dim,
                self.global_aggregator_config.output_dim,
                pos,
            )
        if isinstance(self.regression_head_config, Sequence):
            for head in self.regression_head_config:
                if head.input_dimensions is None:
                    head.input_dimensions = self.global_aggregator_config.output_dim

        return self


def _fill_encoder_dims(
    encoder_config: EncoderConfig,
    embed_dim: int,
    pos: RelativeDistancePositionalEncodingConfig,
):
    attn = encoder_config.attention_layer_config
    if attn.embedding_dim is None:
        attn.embedding_dim = embed_dim
    if encoder_config.d_pair is None:
        encoder_config.d_pair = pos.d_projection
    if encoder_config.d_geo is None:
        encoder_config.d_geo = pos.N_radial_basis_functions


def _fill_aggregator_dims(agg: GlobalAggregatorConfig, embed_dim: int | None):
    if agg.input_dim is None:
        agg.input_dim = embed_dim
    if agg.output_dim is None:
        agg.output_dim = agg.input_dim


def _fill_decoder_dims(
    decoder_config: DecoderConfig,
    embed_dim: int,
    descriptor_dim: int | None,
    pos: RelativeDistancePositionalEncodingConfig,
):
    attn = decoder_config.attention_layer_config
    if attn.embedding_dim is None:
        attn.embedding_dim = embed_dim
    if decoder_config.d_descriptor is None:
        decoder_config.d_descriptor = descriptor_dim
    if decoder_config.d_pair is None:
        decoder_config.d_pair = pos.d_projection
    if decoder_config.d_geo is None:
        decoder_config.d_geo = pos.N_radial_basis_functions
