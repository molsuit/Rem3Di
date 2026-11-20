"""Building a ``MaceFeaturizer`` from a loaded MACE model.

The model is rebuilt from its configuration without e3nn's TorchScript code generation
(Dynamo cannot trace TorchScript modules), optionally converted to cuequivariance, and
traced into the mace-torch fork's simplified view (``mace.tools.compile``), the
preparation its own compiled training and inference use.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch
from e3nn.nn import FullyConnectedNet
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
)
from mace.modules.models import MACE
from mace.modules.polar.model import PolarMACE
from mace.tools.compile import disable_e3nn_codegen, simplified_view
from mace.tools.scripts_utils import extract_config_mace_model
from torch import nn

from remedi.model.featurizer.mace_featurizer import MaceFeaturizer
from remedi.utils.model_utils import mace_per_layer_irreps

# Interactions that multiply their convolution by the cutoff passed to them: with a
# radial embedding that returns the cutoff separately (apply_cutoff=False), a pair
# beyond r_max carries nothing.
_CUTOFF_MULTIPLYING_INTERACTIONS = (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
)
# Interactions whose convolution weights are a function of the radial features alone:
# with the cutoff folded into those features (apply_cutoff=True), a pair beyond r_max
# has zero radial features, and a bias-free network maps them to zero weights.
_RADIAL_ONLY_INTERACTIONS = (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)


def permutes_edge_vectors(model: nn.Module) -> bool:
    """Whether ``model`` feeds its edge vectors to the spherical harmonics as
    ``(y, z, x)``: PolarMACE does, MACE and ScaleShiftMACE do not. Other model
    classes are not supported."""
    if isinstance(model, PolarMACE):
        return True
    if isinstance(model, MACE):
        return False
    raise ValueError(f"the featurizer does not support {type(model).__name__} models")


def check_distant_pairs_are_inert(model: nn.Module) -> None:
    """Raise unless every pair beyond the model's cutoff carries exactly nothing.

    The featurizer's graph holds every pair within a molecule, so its features equal
    the model's only when those beyond the cutoff (and the padding pairs, placed
    beyond it) add nothing to any message, density or weight.
    """
    if hasattr(model, "joint_embedding"):
        raise ValueError(
            "the model embeds the total charge and spin into its node features; "
            "the featurizer does not supply them"
        )
    cutoff_returned_separately = (
        getattr(model.radial_embedding, "apply_cutoff", True) is False
    )
    for index, interaction in enumerate(model.interactions):
        name = type(interaction).__name__
        if cutoff_returned_separately:
            if not isinstance(interaction, _CUTOFF_MULTIPLYING_INTERACTIONS):
                raise ValueError(
                    f"interaction {index} ({name}) is not known to apply the cutoff "
                    "it is given: pairs beyond the cutoff may contribute"
                )
        elif not isinstance(interaction, _RADIAL_ONLY_INTERACTIONS) or not isinstance(
            interaction.conv_tp_weights, FullyConnectedNet
        ):
            raise ValueError(
                f"interaction {index} ({name}) with the cutoff folded into the radial "
                "features: its weights are not known to vanish beyond the cutoff"
            )


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Generator[None]:
    """Build modules in ``dtype``: a model rebuilt in float32 and cast afterwards would
    have lost the float64 checkpoint's precision in the state-dict copy."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _rebuild_without_codegen(model: nn.Module) -> nn.Module:
    """``model`` rebuilt from its configuration (e3nn code generation is off around
    the call) with its weights.

    The state may differ only in state that is not trained: the rebuilt model may
    have the symmetric contractions' ``*_zeroed`` flags (set from the configuration
    on construction; checkpoints from before they existed lack them), and the loaded
    one may have the weightless elementwise products of e3nn gates (state of their
    TorchScript modules).
    """
    rebuilt = model.__class__(**extract_config_mace_model(model))
    result = rebuilt.load_state_dict(model.state_dict(), strict=False)
    missing = [key for key in result.missing_keys if not key.endswith("_zeroed")]
    if missing:
        raise RuntimeError(f"rebuilt MACE model misses weights {missing}")
    source_state = model.state_dict()
    for key in result.unexpected_keys:
        weightless_gate_state = ".equivariant_nonlin.mul." in key and (
            not key.endswith(".weight") or source_state[key].numel() == 0
        )
        if not weightless_gate_state:
            raise RuntimeError(f"rebuilt MACE model has no place for {key}")
    return rebuilt


def build_mace_featurizer(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    enable_cueq: bool,
) -> MaceFeaturizer:
    """The featurizer of ``model`` (a loaded MACE model, left unchanged) on
    ``device`` in ``dtype``, with cuequivariance kernels if ``enable_cueq`` (CUDA
    only). It runs eagerly; ``MaceFeaturizer.compile_features`` compiles it."""
    if enable_cueq and device.type != "cuda":
        raise ValueError("cuequivariance kernels need a CUDA device")
    check_distant_pairs_are_inert(model)
    permute_edge_vectors = permutes_edge_vectors(model)
    per_layer_irreps = mace_per_layer_irreps(model)
    with disable_e3nn_codegen(), _default_dtype(dtype):
        if enable_cueq:
            from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

            prepared = run_e3nn_to_cueq(model, device=device.type, layout="ir_mul")
        else:
            prepared = _rebuild_without_codegen(model)
    prepared = prepared.to(device=device, dtype=dtype).eval()
    prepared.requires_grad_(False)
    return MaceFeaturizer(
        simplified_view(prepared),
        per_layer_irreps=per_layer_irreps,
        layout="ir_mul" if enable_cueq else "mul_ir",
        permute_edge_vectors=permute_edge_vectors,
    ).eval()
