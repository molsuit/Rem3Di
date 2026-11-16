from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

import torch
import torch.nn as nn
import yaml
from e3nn.o3 import Irreps
from pydantic import ConfigDict, Field, model_validator

from remedi.configuration.config_utils import IrrepType, StrictBaseModel
from remedi.configuration.mace_config import MaceConfig
from remedi.model.pooling import AttnPool, MeanPool, PMAAggregator
from remedi.model.preprocessing.radial_basis_functions import (
    BesselBasisFunctions,
    GaussianBasisFunctions,
)
from remedi.utils.model_utils import get_invariant_indices

if TYPE_CHECKING:
    from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
        AtomicDescriptorPreprocessor,
    )
    from remedi.model.preprocessing.geometric_preprocessor import (
        PairDistanceMatrixGeometricPreprocessor,
    )
    from remedi.model.preprocessing.preprocessing import Preprocessor
    from remedi.model.regression_models import MultiTaskRegressionModel
    from remedi.model.remedi_model import REM3DIModel


class HeadType(Enum):
    RESIDUAL = "residual"
    FULLY_CONNECTED = "fully_connected"
    LINEAR = "linear"


ActivationName = Literal["silu", "gelu", "relu", "tanh"]

ACTIVATION_MODULES: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def build_activation(name: ActivationName) -> nn.Module:
    """A fresh activation module, so no state is shared between blocks."""
    return ACTIVATION_MODULES[name]()


@dataclass(frozen=True)
class ModelDimensions:
    """Widths every part of the model derives from the architecture config.

    None of these is settable in yaml: each one is fixed by another field, and
    letting the yaml repeat it only allowed mismatches that failed in forward.
    """

    embedding_dimension: int  # atomic embedding width after preprocessing
    pair_dimension: int  # pair representation width (d_projection)
    radial_basis_dimension: int  # number of radial basis functions
    descriptor_dimension: int  # width of one descriptor token
    descriptor_sequence_length: int  # number of descriptor tokens

    @property
    def descriptor_flat_dimension(self) -> int:
        return self.descriptor_sequence_length * self.descriptor_dimension


class AttentionLayerConfig(StrictBaseModel):
    num_heads: int
    dim_feedforward: int
    dropout: float


class EncoderConfig(StrictBaseModel):
    N_layers: int
    attention_layer_config: AttentionLayerConfig

    def build(
        self, global_aggregator: nn.Module, dimensions: ModelDimensions
    ) -> nn.Module:
        from remedi.model.pair_encoder import TransformerPairEncoder

        return TransformerPairEncoder(self, global_aggregator, dimensions)


class DecoderConfig(StrictBaseModel):
    N_layers: int
    attention_layer_config: AttentionLayerConfig

    def build(self, dimensions: ModelDimensions) -> nn.Module:
        from remedi.model.decoder import TransformerPairDecoder

        return TransformerPairDecoder(self, dimensions)


class RegressionHeadConfig(StrictBaseModel):
    task_name: str
    activation: ActivationName = "silu"
    hidden_dimensions: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    # Width of the per-task auxiliary data concatenated to the flat descriptor
    # (``Sample.auxillary_data[task_name]``); 0 when the task has none.
    auxiliary_input_dimension: int = 0
    head_type: HeadType = HeadType.FULLY_CONNECTED
    # ``None`` -> scalar regression head (final Linear -> 1).
    # A positive int -> single-label classification head over that many classes
    # (final Linear -> n_classes, raw logits). This is the only thing that
    # distinguishes a regression from a classification architecture; everything
    # upstream (preprocessor / encoder / aggregator) is shared.
    n_classes: int | None = None

    @model_validator(mode="after")
    def _linear_head_has_no_hidden_layers(self) -> Self:
        if self.head_type == HeadType.LINEAR and self.hidden_dimensions:
            raise ValueError(
                "head_type 'linear' is a single Linear layer; set "
                "hidden_dimensions to [] instead of "
                f"{self.hidden_dimensions}."
            )
        return self


class PrecomputedInvariantNormalizationConfig(StrictBaseModel):
    kind: Literal["precomputed_normalization"] = "precomputed_normalization"

    def build(self, invariant_dimension: int) -> nn.Module:
        from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
            PrecomputedInvariantNormalization,
        )

        return PrecomputedInvariantNormalization(
            invariant_dimension=invariant_dimension
        )


class OnTheFlyInvariantNormalizationConfig(StrictBaseModel):
    kind: Literal["on_the_fly_normalization"] = "on_the_fly_normalization"
    momentum: float
    warm_up_batches: int

    def build(self, invariant_dimension: int) -> nn.Module:
        from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
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


class IdentityInvariantProjectionConfig(StrictBaseModel):
    """Pass invariants through unchanged."""

    kind: Literal["identity"] = "identity"

    def output_dim_for(self, input_dim: int) -> int:
        return input_dim

    def build(self, input_dim: int) -> nn.Module:
        return nn.Identity()


class LinearInvariantProjectionConfig(StrictBaseModel):
    """Single linear down (or up) projection applied atomwise to the invariants."""

    kind: Literal["linear"] = "linear"
    output_dim: int
    bias: bool = True

    def output_dim_for(self, input_dim: int) -> int:
        return self.output_dim

    def build(self, input_dim: int) -> nn.Module:
        return nn.Linear(input_dim, self.output_dim, bias=self.bias)


class MLPInvariantProjectionConfig(StrictBaseModel):
    """Multi-layer perceptron projection applied atomwise to the invariants."""

    kind: Literal["mlp"] = "mlp"
    output_dim: int
    hidden_dimensions: list[int]
    activation: ActivationName = "silu"
    dropout: float = 0.0

    def output_dim_for(self, input_dim: int) -> int:
        return self.output_dim

    def build(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        previous_dimension = input_dim
        for hidden_dimension in self.hidden_dimensions:
            layers.append(nn.Linear(previous_dimension, hidden_dimension))
            layers.append(build_activation(self.activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous_dimension = hidden_dimension
        layers.append(nn.Linear(previous_dimension, self.output_dim))
        return nn.Sequential(*layers)


InvariantProjectionConfig = Annotated[
    IdentityInvariantProjectionConfig
    | LinearInvariantProjectionConfig
    | MLPInvariantProjectionConfig,
    Field(discriminator="kind"),
]


class EmbeddingPreprocessConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Irreps emitted by each MACE message-passing layer (product stack), in
    # order. Filled from the architecture's MaceConfig (and checked against it
    # when given); required when the model consumes precomputed embeddings.
    # The full input irreps are their concatenation (`input_irreps`).
    mace_per_layer_irreps: list[IrrepType] | None = None
    # Subset of MACE message-passing layers to use. None = all layers.
    mace_layer_indices: list[int] | None = None
    pseudoscalar_dimension: int
    chiral_embedding_dimension: int
    gated: bool = True
    # Chiral (pseudoscalar) embedding of the equivariant features, preceded by
    # an RMS norm of the l=1 blocks. Off: only the invariants are used.
    pseudoscalars: bool = True
    invariant_normalization_config: InvNormConfig = (
        PrecomputedInvariantNormalizationConfig()
    )
    invariant_projection_config: InvariantProjectionConfig = (
        IdentityInvariantProjectionConfig()
    )

    def check_layer_selection(self) -> None:
        """Validate `mace_layer_indices` against the per-layer irreps, once known."""
        if self.mace_layer_indices is None or self.mace_per_layer_irreps is None:
            return
        number_of_layers = len(self.mace_per_layer_irreps)
        for index in self.mace_layer_indices:
            if index < 0 or index >= number_of_layers:
                raise ValueError(
                    f"mace_layer_indices contains {index}, out of range "
                    f"[0, {number_of_layers}) for the configured MACE layer stack."
                )
        if len(set(self.mace_layer_indices)) != len(self.mace_layer_indices):
            raise ValueError("mace_layer_indices must not contain duplicates.")
        if self.mace_layer_indices != sorted(self.mace_layer_indices):
            raise ValueError(
                "mace_layer_indices must be sorted ascending so the channel "
                "slice preserves layer order."
            )

    @model_validator(mode="after")
    def _validate_layer_selection(self) -> Self:
        self.check_layer_selection()
        return self

    @property
    def per_layer_irreps(self) -> list[Irreps]:
        if self.mace_per_layer_irreps is None:
            raise ValueError(
                "embedding_preprocess_config.mace_per_layer_irreps is not set: "
                "give a mace_config to derive it from, or set it explicitly."
            )
        return self.mace_per_layer_irreps

    @property
    def input_irreps(self) -> Irreps:
        """Irreps of the full MACE node features (all layers concatenated)."""
        irreps = Irreps()
        for layer_irreps in self.per_layer_irreps:
            irreps = irreps + layer_irreps
        return irreps

    @property
    def selected_layer_indices(self) -> list[int]:
        if self.mace_layer_indices is None:
            return list(range(len(self.per_layer_irreps)))
        return list(self.mace_layer_indices)

    @property
    def selected_input_irreps(self) -> Irreps:
        """Concatenation of the per-layer irreps for the selected layers."""
        irreps = Irreps()
        for index in self.selected_layer_indices:
            irreps = irreps + self.per_layer_irreps[index]
        return irreps

    @property
    def selected_invariant_irreps(self) -> Irreps:
        _, invariants = get_invariant_indices(self.selected_input_irreps)
        return invariants

    @property
    def selected_invariant_dimension(self) -> int:
        return self.selected_invariant_irreps.dim

    @property
    def selected_channel_indices(self) -> list[int] | None:
        """Channel positions in `input_irreps` that correspond to the selected
        layers. Returns None when no slicing is needed (selection covers all
        layers contiguously from the start)."""
        indices = self.selected_layer_indices
        if indices == list(range(len(self.per_layer_irreps))):
            return None
        cursor = 0
        ranges: list[tuple[int, int]] = []
        for layer_irreps in self.per_layer_irreps:
            ranges.append((cursor, cursor + layer_irreps.dim))
            cursor += layer_irreps.dim
        return [channel for index in indices for channel in range(*ranges[index])]

    @property
    def projected_invariant_dimension(self) -> int:
        return self.invariant_projection_config.output_dim_for(
            self.selected_invariant_dimension
        )

    @property
    def output_irreps(self) -> Irreps:
        # Output is purely l=0: the (projected) even invariants, then the odd
        # chiral embedding. The e3nn tuples are (l, parity): (0, 1) = 0e,
        # (0, -1) = 0o. MACE's l>=1 channels are consumed internally by the
        # chiral embedding model. After projection the invariants no longer
        # follow MACE's invariant block structure, so they form one 0e block.
        even_invariants = Irreps([(self.projected_invariant_dimension, (0, 1))])
        if not self.pseudoscalars:
            return even_invariants
        return even_invariants + Irreps([(self.chiral_embedding_dimension, (0, -1))])

    @property
    def output_irreps_dim(self) -> int:
        return self.output_irreps.dim

    @property
    def invariant_irreps(self) -> Irreps:
        """Invariant irreps of the full (unselected) input."""
        _, irreps = get_invariant_indices(self.input_irreps)
        return irreps

    def _slice_stats_to_selection(self, stats: torch.Tensor) -> torch.Tensor:
        """Slice precomputed (full-invariant) stats down to the channels
        corresponding to the selected layers' invariants.

        Accepts the user-supplied tensor sized either to
        `selected_invariant_dimension` (already sliced) or to
        `invariant_irreps.dim` (full invariants — gets sliced here).
        """
        full_dimension = self.invariant_irreps.dim
        selected_dimension = self.selected_invariant_dimension
        last = stats.shape[-1]
        if last == selected_dimension:
            return stats
        if last != full_dimension:
            raise ValueError(
                f"Precomputed stats trailing dim {last} matches neither the "
                f"selected invariant dim ({selected_dimension}) nor the full "
                f"invariant dim ({full_dimension})."
            )

        # Build the slice indices in *invariant-only* coordinates.
        cursor = 0
        layer_invariant_ranges: list[tuple[int, int]] = []
        for layer_irreps in self.per_layer_irreps:
            _, layer_invariants = get_invariant_indices(layer_irreps)
            layer_invariant_ranges.append((cursor, cursor + layer_invariants.dim))
            cursor += layer_invariants.dim
        selected = [
            channel
            for index in self.selected_layer_indices
            for channel in range(*layer_invariant_ranges[index])
        ]
        index_tensor = torch.tensor(selected, dtype=torch.long, device=stats.device)
        return stats.index_select(-1, index_tensor)

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "AtomicDescriptorPreprocessor":
        from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
            AtomicDescriptorPreprocessor,
            PrecomputedInvariantNormalization,
        )

        selected_invariant_dimension = self.selected_invariant_dimension
        invariant_normalization = self.invariant_normalization_config.build(
            invariant_dimension=selected_invariant_dimension
        )

        if (
            isinstance(invariant_normalization, PrecomputedInvariantNormalization)
            and mean_atomic_embedding is not None
            and std_atomic_embedding is not None
        ):
            invariant_normalization.set_stats(
                mean=self._slice_stats_to_selection(mean_atomic_embedding),
                std=self._slice_stats_to_selection(std_atomic_embedding),
            )

        invariant_projection = self.invariant_projection_config.build(
            input_dim=selected_invariant_dimension
        )

        return AtomicDescriptorPreprocessor(
            preprocess_config=self,
            invariant_normalization=invariant_normalization,
            invariant_projection=invariant_projection,
            selected_channel_indices=self.selected_channel_indices,
        )


class MeanAggregatorConfig(StrictBaseModel):
    aggregator_type: Literal["mean"] = "mean"

    @property
    def descriptor_seq_len(self) -> int:
        return 1

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        return MeanPool(d_in=input_dim, d_out=output_dim, output_dropout=output_dropout)


class AttentionAggregatorConfig(StrictBaseModel):
    """Multi-head attention pooling with one learned query per head.

    `head_dim` is the per-head key width (as for `PMAAggregatorConfig`); the
    total key width is `num_heads * head_dim`. None: `input_dim / num_heads`.
    """

    aggregator_type: Literal["attention"] = "attention"
    num_heads: int
    head_dim: int | None = None
    attn_dropout: float | None = None

    @property
    def descriptor_seq_len(self) -> int:
        return 1

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        return AttnPool(
            d_in=input_dim,
            d_out=output_dim,
            d_hidden=None if self.head_dim is None else self.num_heads * self.head_dim,
            n_heads=self.num_heads,
            dropout=self.attn_dropout or 0.0,
            output_dropout=output_dropout,
        )


class PMAAggregatorConfig(StrictBaseModel):
    """Set Transformer pooling-by-multihead-attention.

    `head_dim` is the per-head Q/K dim (PyTorch convention); the total Q/K
    dim is `num_heads * head_dim`. `num_seeds` learnable queries cross-attend
    to the input set; each seed produces an `output_dim`-dim vector and the
    aggregator returns the full `(B, num_seeds, output_dim)` sequence so the
    decoder can cross-attend to each seed independently. `output_dim` must
    be divisible by `num_heads`.
    """

    aggregator_type: Literal["pma_attention"] = "pma_attention"
    head_dim: int
    num_heads: int = 4
    attn_dropout: float = 0.0
    num_seeds: int = 16
    use_mlp: bool = False

    @property
    def descriptor_seq_len(self) -> int:
        return self.num_seeds

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


class GlobalAggregatorConfig(StrictBaseModel):
    aggregator_type_config: AggUnion
    # Width of one descriptor token. None: the atomic embedding width.
    output_dim: int | None = None
    global_molecular_descriptor_dropout: float | None = None

    @property
    def descriptor_seq_len(self) -> int:
        return self.aggregator_type_config.descriptor_seq_len

    def build(self, dimensions: ModelDimensions) -> nn.Module:
        return self.aggregator_type_config.build(
            input_dim=dimensions.embedding_dimension,
            output_dim=dimensions.descriptor_dimension,
            output_dropout=self.global_molecular_descriptor_dropout,
        )


class GaussianBasisConfig(StrictBaseModel):
    kind: Literal["gaussian"] = "gaussian"

    def build(self, N_radial_basis_functions: int, distance_cutoff: float) -> nn.Module:
        return GaussianBasisFunctions(N_radial_basis_functions, distance_cutoff)


class BesselBasisConfig(StrictBaseModel):
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


class RelativeDistancePositionalEncodingConfig(StrictBaseModel):
    N_radial_basis_functions: int
    distance_cutoff: float
    d_projection: int
    basis_function_config: BasisFunctionConfig = GaussianBasisConfig()

    def build(self) -> "PairDistanceMatrixGeometricPreprocessor":
        from remedi.model.preprocessing.geometric_preprocessor import (
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


class _BaseArchitectureConfig(StrictBaseModel):
    mace_config: MaceConfig | None = None
    embedding_preprocess_config: EmbeddingPreprocessConfig
    encoder_config: EncoderConfig
    global_aggregator_config: GlobalAggregatorConfig
    positional_encoding_config: RelativeDistancePositionalEncodingConfig

    @model_validator(mode="after")
    def _resolve_mace_per_layer_irreps(self) -> Self:
        embedding_config = self.embedding_preprocess_config
        if self.mace_config is not None:
            mace_irreps = self.mace_config.get_per_layer_irreps()
            if embedding_config.mace_per_layer_irreps is None:
                embedding_config.mace_per_layer_irreps = mace_irreps
            elif list(embedding_config.mace_per_layer_irreps) != mace_irreps:
                raise ValueError(
                    "embedding_preprocess_config.mace_per_layer_irreps "
                    f"{[str(irreps) for irreps in embedding_config.mace_per_layer_irreps]} "
                    "does not match the configured MACE model "
                    f"{[str(irreps) for irreps in mace_irreps]}."
                )
        elif embedding_config.mace_per_layer_irreps is None:
            raise ValueError(
                "embedding_preprocess_config.mace_per_layer_irreps is not set and "
                "no mace_config was provided to derive it from."
            )
        embedding_config.check_layer_selection()
        return self

    @property
    def dimensions(self) -> ModelDimensions:
        embedding_dimension = self.embedding_preprocess_config.output_irreps_dim
        output_dim = self.global_aggregator_config.output_dim
        return ModelDimensions(
            embedding_dimension=embedding_dimension,
            pair_dimension=self.positional_encoding_config.d_projection,
            radial_basis_dimension=(
                self.positional_encoding_config.N_radial_basis_functions
            ),
            descriptor_dimension=(
                embedding_dimension if output_dim is None else output_dim
            ),
            descriptor_sequence_length=self.global_aggregator_config.descriptor_seq_len,
        )

    def _build_mace_model(self):
        if self.mace_config is None:
            return None
        return self.mace_config.build_torch_sim_model()

    def _build_preprocessor(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "Preprocessor":
        """Atomic preprocessor in float64, geometric preprocessor in float32.

        The MACE model keeps the dtype its MaceConfig asks for: casting it
        would desynchronise it from the torch-sim wrapper's own dtype.
        """
        from remedi.model.preprocessing.preprocessing import (
            Preprocessor,
            PreprocessorWithAtomicEmbedding,
        )

        atomic = self.embedding_preprocess_config.build(
            mean_atomic_embedding, std_atomic_embedding
        ).double()
        geometric = self.positional_encoding_config.build().float()
        mace_model = self._build_mace_model()
        if mace_model is not None:
            return PreprocessorWithAtomicEmbedding(
                mace_model=mace_model,
                atomic_preprocessor=atomic,
                geometric_preprocessor=geometric,
                node_feature_irreps=self.embedding_preprocess_config.input_irreps,
            )
        return Preprocessor(
            atomic_preprocessor=atomic, geometric_preprocessor=geometric
        )

    def _build_encoder(self) -> nn.Module:
        dimensions = self.dimensions
        aggregator = self.global_aggregator_config.build(dimensions)
        return self.encoder_config.build(aggregator, dimensions).float()


class EncoderOnlyArchitectureConfig(_BaseArchitectureConfig):
    kind: Literal["encoder_only"] = "encoder_only"

    @classmethod
    def from_encoder_yaml(
        cls,
        directory: str | Path,
        trained: bool = True,
        mace_device: str | None = None,
        mace_model_path: Path | None = None,
    ) -> Self:
        """Load encoder + preprocessor from a checkpoint, ignoring decoder fields.

        Pretraining writes ``kind: encoder_decoder`` configs; descriptor probes
        only need the encoder half. This reads the yaml as a dict, strips the
        decoder block, and forces ``kind: encoder_only`` so the same checkpoint
        directory can be consumed without staging a patched config on disk.

        ``mace_device`` / ``mace_model_path`` replace the training-time MACE
        device and (absolute, machine-specific) model path before validation,
        which already loads MACE to read its irreps.
        """
        filename = (
            "post_training_architecture_config.yaml"
            if trained
            else "architecture_config.yaml"
        )
        with open(Path(directory) / filename) as fh:
            data = yaml.safe_load(fh)
        data.pop("decoder_config", None)
        data["kind"] = "encoder_only"
        if mace_device is not None or mace_model_path is not None:
            mace_data = data.get("mace_config")
            if mace_data is None:
                raise ValueError(
                    f"{Path(directory) / filename} has no mace_config to apply "
                    "the MACE device / model path override to."
                )
            if mace_device is not None:
                mace_data["device"] = mace_device
            if mace_model_path is not None:
                mace_data["model_path"] = str(mace_model_path)
        return cls.model_validate(data)

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "REM3DIModel":
        from remedi.model.remedi_model import REM3DIModel

        if self.mace_config is None:
            raise ValueError("EncoderOnlyArchitectureConfig.build requires mace_config")
        return REM3DIModel(
            preprocessor=self._build_preprocessor(
                mean_atomic_embedding, std_atomic_embedding
            ),
            encoder=self._build_encoder(),
        )


class EncoderDecoderArchitectureConfig(_BaseArchitectureConfig):
    kind: Literal["encoder_decoder"] = "encoder_decoder"
    decoder_config: DecoderConfig

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> EncoderDecoderBundle:
        if self.mace_config is None:
            raise ValueError(
                "EncoderDecoderArchitectureConfig.build requires mace_config"
            )
        return EncoderDecoderBundle(
            preprocessor=self._build_preprocessor(
                mean_atomic_embedding, std_atomic_embedding
            ),
            encoder=self._build_encoder(),
            decoder=self.decoder_config.build(self.dimensions).float(),
        )


class RegressionArchitectureConfig(_BaseArchitectureConfig):
    kind: Literal["regression"] = "regression"
    regression_head_config: list[RegressionHeadConfig]

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "MultiTaskRegressionModel":
        """With a ``mace_config`` the preprocessor runs MACE on the fly from raw
        atoms (supervised training from scratch — the only path that trains the
        pseudoscalar preprocessor). Without one the model consumes precomputed
        ``Sample.embeddings`` (described by ``mace_per_layer_irreps``)."""
        from remedi.model.regression_models import (
            MultitaskHeads,
            MultiTaskRegressionModel,
        )

        heads = MultitaskHeads(
            regression_head_configs=self.regression_head_config,
            descriptor_flat_dimension=self.dimensions.descriptor_flat_dimension,
        ).float()
        return MultiTaskRegressionModel(
            preprocessor=self._build_preprocessor(
                mean_atomic_embedding, std_atomic_embedding
            ),
            encoder=self._build_encoder(),
            regression_heads=heads,
        )


ArchitectureConfig = Annotated[
    EncoderOnlyArchitectureConfig
    | EncoderDecoderArchitectureConfig
    | RegressionArchitectureConfig,
    Field(discriminator="kind"),
]
