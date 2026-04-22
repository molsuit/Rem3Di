import importlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal, Self

import pydantic_yaml as pyaml
import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    computed_field,
    model_validator,
)

from threedscriptors.configuration.config_utils import IrrepType
from threedscriptors.configuration.mace_config import MaceConfig
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
    from mace.calculators import MACECalculator

    from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
        AtomicDescriptorPreprocessor,
    )
    from threedscriptors.model.preprocessing.geometric_preprocessor import (
        PairDistanceMatrixGeometricPreprocessor,
    )
    from threedscriptors.model.preprocessing.preprocessing import Preprocessor
    from threedscriptors.model.regression_models import MultiTaskRegressionModel
    from threedscriptors.model.remedi_model import REM3DIModel


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


def _to_activation_fn(v: str | Callable | Activations) -> Callable:
    if isinstance(v, Activations):
        return Activations.get_activation_fn(v.value.upper())
    if isinstance(v, str):
        return Activations.get_activation_fn(v.upper())
    if callable(v):
        return v
    raise ValueError(f"Cannot coerce {v!r} to an activation function.")


ActivationFn = Annotated[
    Callable,
    BeforeValidator(_to_activation_fn),
    PlainSerializer(lambda fn: fn.__class__.__name__, return_type=str),
]


class AttentionLayerConfig(BaseModel):
    num_heads: int
    dim_feedforward: int
    dropout: float
    embedding_dim: int | None = None


class EncoderConfig(BaseModel):
    N_layers: int
    attention_layer_config: AttentionLayerConfig
    d_pair: int | None = None
    d_geo: int | None = None

    def build(self, global_aggregator: nn.Module) -> nn.Module:
        from threedscriptors.model.pair_encoder import TransformerPairEncoder

        return TransformerPairEncoder(self, global_aggregator)


class DecoderConfig(BaseModel):
    N_layers: int
    d_descriptor: int | None = None
    attention_layer_config: AttentionLayerConfig
    d_pair: int | None = None
    d_geo: int | None = None

    def build(self) -> nn.Module:
        from threedscriptors.model.decoder import TransformerPairDecoder

        return TransformerPairDecoder(self)


class RegressionHeadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str | None = None
    task_config: None = None
    activation_fn: ActivationFn = torch.nn.SiLU()
    hidden_dimensions: list[int] = [256, 128]
    input_dimensions: int | None = None
    head_type: HeadType = HeadType.FULLY_CONNECTED


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

    # Resolved from the parent architecture's MaceConfig during cascade; can
    # also be set directly (e.g. in unit tests that don't want to load MACE).
    input_irreps: IrrepType | None = None
    pseudoscalar_dimension: int
    chiral_embedding_dimension: int
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

        return AtomicDescriptorPreprocessor(
            preprocess_config=self,
            invariant_normalization=invariant_normalization,
        )


class MeanAggregatorConfig(BaseModel):
    aggregator_type: Literal["mean"] = "mean"

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        return MeanPool(d_in=input_dim, output_dropout=output_dropout)


class AttentionAggregatorConfig(BaseModel):
    aggregator_type: Literal["attention"] = "attention"
    num_heads: int
    head_dim: int | None = None
    attn_dropout: float | None = None

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        return AttnPool(
            d_in=input_dim,
            d_hidden=self.head_dim,
            n_heads=self.num_heads,
            dropout=self.attn_dropout or 0.0,
            output_dropout=output_dropout,
        )


class PMAAggregatorConfig(BaseModel):
    aggregator_type: Literal["pma_attention"] = "pma_attention"
    head_dim: int
    num_heads: int = 4
    attn_dropout: float = 0.0
    num_seeds: int = 16
    reduction: Literal["mean", "sum", "max"] = "mean"
    use_mlp: bool = False

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        return PMAAggregator(
            d_in=input_dim,
            d_out=output_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            k_seeds=self.num_seeds,
            dropout=self.attn_dropout,
            use_mlp=self.use_mlp,
            output_dropout=output_dropout,
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

    def build(self) -> nn.Module:
        assert (
            self.input_dim is not None and self.output_dim is not None
        ), "Dimensions must be resolved by ArchitectureConfig cascade before build."

        return self.aggregator_type_config.build(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            output_dropout=self.global_molecular_descriptor_dropout,
        )


class GaussianBasisConfig(BaseModel):
    kind: Literal["gaussian"] = "gaussian"

    def build(self, N_radial_basis_functions: int, distance_cutoff: float) -> nn.Module:
        return GaussianBasisFunctions(N_radial_basis_functions, distance_cutoff)


class BesselBasisConfig(BaseModel):
    kind: Literal["bessel"] = "bessel"
    eps: float = 1e-8

    def build(self, N_radial_basis_functions: int, distance_cutoff: float) -> nn.Module:
        return BesselBasisFunctions(
            N_radial_basis_functions, distance_cutoff, eps=self.eps
        )


BasisFunctionConfig = Annotated[
    GaussianBasisConfig | BesselBasisConfig,
    Field(discriminator="kind"),
]


class RelativeDistancePositionalEncodingConfig(BaseModel):
    N_radial_basis_functions: int
    distance_cutoff: float
    d_projection: int
    basis_function_config: BasisFunctionConfig = GaussianBasisConfig()

    def build(self) -> "PairDistanceMatrixGeometricPreprocessor":
        from threedscriptors.model.preprocessing.geometric_preprocessor import (
            PairDistanceMatrixGeometricPreprocessor,
        )

        radial_basis = self.basis_function_config.build(
            N_radial_basis_functions=self.N_radial_basis_functions,
            distance_cutoff=self.distance_cutoff,
        )
        return PairDistanceMatrixGeometricPreprocessor(
            radial_basis=radial_basis,
            N_radial_basis_functions=self.N_radial_basis_functions,
            distance_cutoff=self.distance_cutoff,
            d_projection=self.d_projection,
        )


@dataclass
class EncoderDecoderBundle:
    preprocessor: "Preprocessor"
    encoder: nn.Module
    decoder: nn.Module


class _BaseArchitectureConfig(BaseModel):
    mace_config: MaceConfig | None = None
    embedding_preprocess_config: EmbeddingPreprocessConfig
    encoder_config: EncoderConfig
    global_aggregator_config: GlobalAggregatorConfig
    positional_encoding_config: RelativeDistancePositionalEncodingConfig

    @model_validator(mode="after")
    def _cascade_shared_dims(self):
        if self.embedding_preprocess_config.input_irreps is None:
            if self.mace_config is None:
                raise ValueError(
                    "embedding_preprocess_config.input_irreps is not set and "
                    "no mace_config was provided to derive it from."
                )
            self.embedding_preprocess_config.input_irreps = (
                self.mace_config.get_irrep_signature()
            )

        embed_dim = self.embedding_preprocess_config.output_irreps_dim
        _fill_encoder_dims(
            self.encoder_config, embed_dim, self.positional_encoding_config
        )
        _fill_aggregator_dims(
            self.global_aggregator_config,
            self.encoder_config.attention_layer_config.embedding_dim,
        )
        return self

    def _build_preprocessor(
        self,
        mace_model=None,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "Preprocessor":
        from threedscriptors.model.preprocessing.preprocessing import (
            Preprocessor,
            PreprocessorWithAtomicEmbedding,
        )

        atomic = self.embedding_preprocess_config.build(
            mean_atomic_embedding, std_atomic_embedding
        )
        geometric = self.positional_encoding_config.build()
        if mace_model is not None:
            return PreprocessorWithAtomicEmbedding(
                mace_model=mace_model,
                atomic_preprocessor=atomic,
                geometric_preprocessor=geometric,
            )
        return Preprocessor(
            atomic_preprocessor=atomic, geometric_preprocessor=geometric
        )

    def _build_encoder(self) -> nn.Module:
        aggregator = self.global_aggregator_config.build()
        return self.encoder_config.build(aggregator)

    @classmethod
    def from_directory(cls, directory: str, trained: bool = True) -> Self:
        filename = (
            "post_training_architecture_config.yaml"
            if trained
            else "architecture_config.yaml"
        )
        return pyaml.parse_yaml_file_as(cls, f"{directory}/{filename}")


class EncoderOnlyArchitectureConfig(_BaseArchitectureConfig):
    kind: Literal["encoder_only"] = "encoder_only"

    def build(
        self,
        mace_calculator: "MACECalculator | None" = None,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "REM3DIModel":
        from threedscriptors.model.remedi_model import REM3DIModel

        if self.mace_config is None:
            raise ValueError("EncoderOnlyArchitectureConfig.build requires mace_config")
        mace_model = self.mace_config.build_torch_sim_model()

        preprocessor = self._build_preprocessor(
            mace_model, mean_atomic_embedding, std_atomic_embedding
        )
        encoder = self._build_encoder()
        model = REM3DIModel(
            preprocessor=preprocessor,
            encoder=encoder,
            mace_calculator=mace_calculator,
        ).float()
        model.preprocessor.atomic_preprocessor.double()
        return model


class EncoderDecoderArchitectureConfig(_BaseArchitectureConfig):
    kind: Literal["encoder_decoder"] = "encoder_decoder"
    decoder_config: DecoderConfig

    @model_validator(mode="after")
    def _cascade_decoder_dims(self):
        _fill_decoder_dims(
            self.decoder_config,
            self.embedding_preprocess_config.output_irreps_dim,
            self.global_aggregator_config.output_dim,
            self.positional_encoding_config,
        )
        return self

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> EncoderDecoderBundle:
        if self.mace_config is None:
            raise ValueError(
                "EncoderDecoderArchitectureConfig.build requires mace_config"
            )
        mace_model = self.mace_config.build_torch_sim_model()
        preprocessor = self._build_preprocessor(
            mace_model, mean_atomic_embedding, std_atomic_embedding
        )
        encoder = self._build_encoder()
        decoder = self.decoder_config.build()
        return EncoderDecoderBundle(
            preprocessor=preprocessor, encoder=encoder, decoder=decoder
        )


class RegressionArchitectureConfig(_BaseArchitectureConfig):
    kind: Literal["regression"] = "regression"
    regression_head_config: list[RegressionHeadConfig]

    @model_validator(mode="after")
    def _cascade_head_dims(self):
        out_dim = self.global_aggregator_config.output_dim
        for head in self.regression_head_config:
            if head.input_dimensions is None:
                head.input_dimensions = out_dim
        return self

    def insert_task_configs(self, task_configs) -> None:
        for task_cfg, head_cfg in zip(
            task_configs, self.regression_head_config, strict=True
        ):
            assert head_cfg.task_name == task_cfg.task_name
            head_cfg.task_config = task_cfg

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "MultiTaskRegressionModel":
        from threedscriptors.model.regression_models import (
            MultitaskHeads,
            MultiTaskRegressionModel,
        )

        preprocessor = self._build_preprocessor(
            None, mean_atomic_embedding, std_atomic_embedding
        )
        encoder = self._build_encoder()
        model = MultiTaskRegressionModel(
            preprocessor=preprocessor,
            encoder=encoder,
            regression_heads=MultitaskHeads(
                regression_head_configs=self.regression_head_config
            ),
        ).float()
        model.preprocessor.atomic_preprocessor.double()
        return model


ArchitectureConfig = Annotated[
    EncoderOnlyArchitectureConfig
    | EncoderDecoderArchitectureConfig
    | RegressionArchitectureConfig,
    Field(discriminator="kind"),
]


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
