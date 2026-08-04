import importlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

import pydantic_yaml as pyaml
import torch
import torch.nn as nn
import yaml
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

from remedi.configuration.config_utils import IrrepType
from remedi.configuration.mace_config import MaceConfig
from remedi.model.pooling_legacy import PMAAggregatorLegacy
from remedi.model.pooling import AttnPool, MeanPool, PMAAggregator
from remedi.model.preprocessing.radial_basis_functions import (
    BesselBasisFunctions,
    GaussianBasisFunctions,
)
from remedi.utils.model_utils import (
    get_equivariant_irreps,
    get_invariant_indices,
)

if TYPE_CHECKING:
    from mace.calculators import MACECalculator

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
        from remedi.model.pair_encoder import TransformerPairEncoder

        return TransformerPairEncoder(self, global_aggregator)


class DecoderConfig(BaseModel):
    N_layers: int
    d_descriptor: int | None = None
    attention_layer_config: AttentionLayerConfig
    d_pair: int | None = None
    d_geo: int | None = None

    def build(self) -> nn.Module:
        from remedi.model.decoder import TransformerPairDecoder

        return TransformerPairDecoder(self)


class RegressionHeadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str | None = None
    task_config: None = None
    activation_fn: ActivationFn = torch.nn.SiLU()
    hidden_dimensions: list[int] = [256, 128]
    input_dimensions: int | None = None
    head_type: HeadType = HeadType.FULLY_CONNECTED
    # ``None`` -> scalar regression head (final Linear -> 1, label scaling).
    # A positive int -> single-label classification head over that many classes
    # (final Linear -> n_classes, raw logits, no label scaling). This is the
    # only thing that distinguishes a regression from a classification
    # architecture; everything upstream (preprocessor / encoder / aggregator) is
    # shared.
    n_classes: int | None = None


class PrecomputedInvariantNormalizationConfig(BaseModel):
    kind: Literal["precomputed_normalization"] = "precomputed_normalization"

    def build(self, invariant_dimension: int) -> nn.Module:
        from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
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


class IdentityInvariantProjectionConfig(BaseModel):
    """Pass invariants through unchanged."""

    kind: Literal["identity"] = "identity"

    def output_dim_for(self, input_dim: int) -> int:
        return input_dim

    def build(self, input_dim: int) -> nn.Module:
        return nn.Identity()


class LinearInvariantProjectionConfig(BaseModel):
    """Single linear down (or up) projection applied atomwise to the invariants."""

    kind: Literal["linear"] = "linear"
    output_dim: int
    bias: bool = True

    def output_dim_for(self, input_dim: int) -> int:
        return self.output_dim

    def build(self, input_dim: int) -> nn.Module:
        return nn.Linear(input_dim, self.output_dim, bias=self.bias)


class MLPInvariantProjectionConfig(BaseModel):
    """Multi-layer perceptron projection applied atomwise to the invariants."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["mlp"] = "mlp"
    output_dim: int
    hidden_dimensions: list[int]
    activation_fn: ActivationFn = torch.nn.SiLU()
    dropout: float = 0.0

    def output_dim_for(self, input_dim: int) -> int:
        return self.output_dim

    def build(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dimensions:
            layers.append(nn.Linear(prev, h))
            # Each block gets its own activation instance so optional state
            # (e.g. learnable parameters in custom activations) is not shared.
            layers.append(type(self.activation_fn)())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        return nn.Sequential(*layers)


InvariantProjectionConfig = Annotated[
    IdentityInvariantProjectionConfig
    | LinearInvariantProjectionConfig
    | MLPInvariantProjectionConfig,
    Field(discriminator="kind"),
]


class EmbeddingPreprocessConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Resolved from the parent architecture's MaceConfig during cascade; can
    # also be set directly (e.g. in unit tests that don't want to load MACE).
    input_irreps: IrrepType | None = None
    # Per-layer breakdown of `input_irreps` (one entry per MACE product stack
    # layer). Required for `mace_layer_indices` to take effect; resolved by the
    # architecture cascade from `MaceConfig.get_per_layer_irreps()` when None.
    mace_per_layer_irreps: list[IrrepType] | None = None
    # Subset of MACE message-passing layers to use. None = all layers.
    mace_layer_indices: list[int] | None = None
    pseudoscalar_dimension: int
    # Load a checkpoint trained before the Rem3DiPseudoScalarTP rewrite. The two
    # modules are not weight-compatible, so this selects the original one.
    legacy_pseudoscalar: bool = False
    chiral_embedding_dimension: int
    gated: bool = True
    pseudoscalars: bool = True
    equivariant_rms_normalization: bool = True
    invariant_normalization_config: InvNormConfig = (
        PrecomputedInvariantNormalizationConfig()
    )
    invariant_projection_config: InvariantProjectionConfig = (
        IdentityInvariantProjectionConfig()
    )

    def _check_layer_selection(self) -> None:
        """Validate `mace_layer_indices` against the per-layer info that is
        currently available. Safe to call multiple times — defers to the cascade
        when per-layer info hasn't been populated yet."""
        if self.mace_layer_indices is None:
            return
        if self.mace_per_layer_irreps is None:
            return
        n_layers = len(self.mace_per_layer_irreps)
        for idx in self.mace_layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(
                    f"mace_layer_indices contains {idx}, out of range "
                    f"[0, {n_layers}) for the configured MACE layer stack."
                )
        if len(set(self.mace_layer_indices)) != len(self.mace_layer_indices):
            raise ValueError("mace_layer_indices must not contain duplicates.")
        if self.mace_layer_indices != sorted(self.mace_layer_indices):
            raise ValueError(
                "mace_layer_indices must be sorted ascending so the channel "
                "slice preserves layer order."
            )

    @model_validator(mode="after")
    def _validate_layer_selection(self) -> "EmbeddingPreprocessConfig":
        self._check_layer_selection()
        return self

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

    @property
    def selected_layer_indices(self) -> list[int] | None:
        """Resolved layer indices, or None when no selection is active.

        Returns None when either `mace_layer_indices` is None (all layers) AND
        per-layer info is missing; in that case the caller treats the input
        irreps as already representing the selection.
        """
        if self.mace_per_layer_irreps is None:
            return None
        if self.mace_layer_indices is None:
            return list(range(len(self.mace_per_layer_irreps)))
        return list(self.mace_layer_indices)

    @property
    def selected_input_irreps(self) -> Irreps:
        """Concatenation of the per-layer irreps for the selected layers."""
        idxs = self.selected_layer_indices
        if idxs is None or self.mace_per_layer_irreps is None:
            assert (
                self.input_irreps is not None
            ), "input_irreps must be resolved before reading selected_input_irreps"
            return self.input_irreps
        out = Irreps()
        for i in idxs:
            out = out + self.mace_per_layer_irreps[i]
        return out

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
        idxs = self.selected_layer_indices
        if idxs is None or self.mace_per_layer_irreps is None:
            return None
        if idxs == list(range(len(self.mace_per_layer_irreps))):
            return None
        cursor = 0
        ranges: list[tuple[int, int]] = []
        for layer_irreps in self.mace_per_layer_irreps:
            ranges.append((cursor, cursor + layer_irreps.dim))
            cursor += layer_irreps.dim
        return [c for i in idxs for c in range(*ranges[i])]

    @property
    def projected_invariant_dimension(self) -> int:
        return self.invariant_projection_config.output_dim_for(
            self.selected_invariant_dimension
        )

    @computed_field(return_type=IrrepType, repr=True)
    @property
    def output_irreps(self):
        # Output is purely l=0 (scalars). The e3nn tuples are (l, parity), not
        # (l, multiplicity): (0, 1) = even scalar (0e), (0, -1) = pseudoscalar
        # (0o). Multiplicities are the leading ints (`projected_invariant_dim`
        # and `chiral_embedding_dimension`). MACE's l>=1 channels are consumed
        # internally — they feed the chiral embedding model, which contracts
        # them down to a pseudoscalar before concatenation here. After
        # projection the invariants no longer correspond to the original MACE
        # invariant block structure, so we represent them as a single 0e block.
        even_invariants = Irreps([(self.projected_invariant_dimension, (0, 1))])

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

    def _slice_stats_to_selection(self, stats: torch.Tensor) -> torch.Tensor:
        """Slice precomputed (full-invariant) stats down to the channels
        corresponding to the selected layers' invariants.

        Accepts the user-supplied tensor sized either to
        `selected_invariant_dimension` (already sliced) or to
        `invariant_irreps.dim` (full invariants — gets sliced here).
        """
        full_dim = self.invariant_irreps.dim
        sel_dim = self.selected_invariant_dimension
        last = stats.shape[-1]
        if last == sel_dim:
            return stats
        if last != full_dim:
            raise ValueError(
                f"Precomputed stats trailing dim {last} matches neither the "
                f"selected invariant dim ({sel_dim}) nor the full invariant "
                f"dim ({full_dim})."
            )
        if self.mace_per_layer_irreps is None:
            return stats  # Nothing we can do without per-layer info.

        # Build the slice indices in *invariant-only* coordinates.
        idxs = self.selected_layer_indices or []
        cursor = 0
        layer_inv_ranges: list[tuple[int, int]] = []
        for layer_irreps in self.mace_per_layer_irreps:
            _, layer_inv = get_invariant_indices(layer_irreps)
            layer_dim = layer_inv.dim
            layer_inv_ranges.append((cursor, cursor + layer_dim))
            cursor += layer_dim
        sel_idx = [c for i in idxs for c in range(*layer_inv_ranges[i])]
        index = torch.tensor(sel_idx, dtype=torch.long, device=stats.device)
        return stats.index_select(-1, index)

    def build(
        self,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "AtomicDescriptorPreprocessor":
        from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
            AtomicDescriptorPreprocessor,
            PrecomputedInvariantNormalization,
        )

        sel_inv_dim = self.selected_invariant_dimension
        invariant_normalization = self.invariant_normalization_config.build(
            invariant_dimension=sel_inv_dim
        )

        if (
            isinstance(invariant_normalization, PrecomputedInvariantNormalization)
            and mean_atomic_embedding is not None
            and std_atomic_embedding is not None
        ):
            mean = self._slice_stats_to_selection(mean_atomic_embedding)
            std = self._slice_stats_to_selection(std_atomic_embedding)
            assert mean.shape[-1] == sel_inv_dim
            invariant_normalization.set_stats(mean=mean, std=std)

        invariant_projection = self.invariant_projection_config.build(
            input_dim=sel_inv_dim
        )

        return AtomicDescriptorPreprocessor(
            preprocess_config=self,
            invariant_normalization=invariant_normalization,
            invariant_projection=invariant_projection,
            selected_channel_indices=self.selected_channel_indices,
        )


class MeanAggregatorConfig(BaseModel):
    aggregator_type: Literal["mean"] = "mean"

    @property
    def descriptor_seq_len(self) -> int:
        return 1

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        return MeanPool(d_in=input_dim, d_out=output_dim, output_dropout=output_dropout)


class AttentionAggregatorConfig(BaseModel):
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
            d_hidden=self.head_dim,
            n_heads=self.num_heads,
            dropout=self.attn_dropout or 0.0,
            output_dropout=output_dropout,
        )


class PMAAggregatorConfig(BaseModel):
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


class LegacyPMAAggregatorConfig(BaseModel):
    """Pre-rewrite PMA, for loading checkpoints trained before the rewrite.

    Differs from :class:`PMAAggregatorConfig` in two ways that are not renames,
    which is why it exists as a separate variant rather than a flag:

    * `head_dim` is the **total** Q/K width across heads, not the per-head
      width. A checkpoint trained with `head_dim: 320` and `num_heads: 8` has
      40-wide heads, where the current config would build 320-wide ones.
    * There is no output projection `W_O`.

    It also reduces over seeds internally, so `descriptor_seq_len` is 1 and the
    descriptor is `(B, output_dim)` rather than `(B, num_seeds, output_dim)`.

    Use `PMAAggregatorConfig` for anything new.
    """

    aggregator_type: Literal["pma_attention_legacy"] = "pma_attention_legacy"
    head_dim: int
    num_heads: int = 4
    attn_dropout: float = 0.0
    num_seeds: int = 16
    use_mlp: bool = False

    @property
    def descriptor_seq_len(self) -> int:
        # The legacy module means over seeds before returning.
        return 1

    def build(
        self, input_dim: int, output_dim: int, output_dropout: float | None = None
    ) -> nn.Module:
        # output_dropout has no counterpart in the pre-rewrite module; accepted
        # for interface parity and deliberately ignored so behaviour is exact.
        del output_dropout
        return PMAAggregatorLegacy(
            d_in=input_dim,
            d_out=output_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            k_seeds=self.num_seeds,
            dropout=self.attn_dropout,
            use_mlp=self.use_mlp,
        )


AggUnion = Annotated[
    MeanAggregatorConfig
    | AttentionAggregatorConfig
    | PMAAggregatorConfig
    | LegacyPMAAggregatorConfig,
    Field(discriminator="aggregator_type"),
]


class GlobalAggregatorConfig(BaseModel):
    aggregator_type_config: AggUnion
    input_dim: int | None = None
    output_dim: int | None = None
    global_molecular_descriptor_dropout: float | None = None

    @computed_field(return_type=int, repr=True)
    @property
    def descriptor_seq_len(self) -> int:
        return self.aggregator_type_config.descriptor_seq_len

    @computed_field(return_type=int | None, repr=True)
    @property
    def descriptor_flat_dim(self) -> int | None:
        if self.output_dim is None:
            return None
        return self.descriptor_seq_len * self.output_dim

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

        if (
            self.embedding_preprocess_config.mace_per_layer_irreps is None
            and self.mace_config is not None
        ):
            self.embedding_preprocess_config.mace_per_layer_irreps = (
                self.mace_config.get_per_layer_irreps()
            )
        # Re-run the layer-selection check now that per-layer info is available.
        self.embedding_preprocess_config._check_layer_selection()

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
        from remedi.model.preprocessing.preprocessing import (
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

    @classmethod
    def from_encoder_yaml(cls, directory: str | Path, trained: bool = True) -> Self:
        """Load encoder + preprocessor from a checkpoint, ignoring decoder fields.

        Pretraining writes ``kind: encoder_decoder`` configs; descriptor probes
        only need the encoder half. This reads the yaml as a dict, strips the
        decoder block, and forces ``kind: encoder_only`` so the same checkpoint
        directory can be consumed without staging a patched config on disk.
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
        return cls.model_validate(data)

    def build(
        self,
        mace_calculator: "MACECalculator | None" = None,
        mean_atomic_embedding: torch.Tensor | None = None,
        std_atomic_embedding: torch.Tensor | None = None,
    ) -> "REM3DIModel":
        from remedi.model.remedi_model import REM3DIModel

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
        # Heads consume the *flattened* descriptor (num_seeds * output_dim for
        # PMA; output_dim for single-token aggregators).
        out_dim = self.global_aggregator_config.descriptor_flat_dim
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
        from remedi.model.regression_models import (
            MultitaskHeads,
            MultiTaskRegressionModel,
        )

        # With a ``mace_config`` the preprocessor runs MACE on the fly from raw
        # atoms (supervised training from scratch — the only path that trains
        # the pseudoscalar preprocessor). Without one the model consumes
        # precomputed ``Sample.embeddings`` (input_irreps given directly).
        mace_model = (
            self.mace_config.build_torch_sim_model()
            if self.mace_config is not None
            else None
        )
        preprocessor = self._build_preprocessor(
            mace_model, mean_atomic_embedding, std_atomic_embedding
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
