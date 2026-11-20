"""torch.compile of the pretraining encoder and decoder with dynamic shapes.

The pair tensors are ``(molecules, atoms, atoms, channels)``. A plain
``nn.Linear`` on them traces the last atom axis as ``(atoms**2) // atoms``,
which made inductor's triton code generation of the backward fail on CUDA
(``CantSplit``); see ``remedi/model/pair_linear.py``. These tests check that no
traced size is a floor division (CPU, no compiler needed), that the compiled
modules agree with eager mode in the outputs and in every parameter gradient,
and that one graph per module serves every batch shape (CPU with gcc >= 10, and
CUDA when available).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from functorch.compile import make_boxed_func
from pydantic import TypeAdapter
from torch import nn
from torch._dynamo.backends.common import aot_autograd

from remedi.configuration.architecture_config import (
    ArchitectureConfig,
    EncoderDecoderArchitectureConfig,
    ModelDimensions,
)
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.model.pair_linear import PairLinear, pair_channel_matmul
from remedi.training.data.sample import PreprocessedSample

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DRUG_LIKE_SMALL = (
    REPOSITORY_ROOT / "configs/training/drug_like_small/architecture_config.yaml"
)

#: (molecules, padded atoms) of the batches every test runs: the padded atom
#: count varies as with ``pad_multiple: 1``. No dimension is 0 or 1, which
#: torch.compile specialises on.
BATCH_SHAPES = [(3, 7), (5, 12), (2, 9), (4, 16), (3, 7)]

#: float32: inductor reorders the reductions.
RELATIVE_TOLERANCE = 1e-4
ABSOLUTE_TOLERANCE = 1e-5

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _architecture(*, small: bool, dropout: float) -> EncoderDecoderArchitectureConfig:
    """The drug_like_small architecture without its MACE model; ``small``
    shrinks every width (and keeps two layers, so a pair update is traced)."""
    raw = yaml.safe_load(DRUG_LIKE_SMALL.read_text())
    raw.pop("mace_config")
    for module in ("encoder_config", "decoder_config"):
        raw[module]["attention_layer_config"]["dropout"] = dropout
    raw["global_aggregator_config"]["aggregator_config"]["attn_dropout"] = dropout
    if small:
        raw["embedding_preprocess_config"]["mace_per_layer_irreps"] = [
            "16x0e+16x1o",
            "16x0e+16x1o",
        ]
        for module in ("encoder_config", "decoder_config"):
            raw[module]["N_layers"] = 2
            raw[module]["embedding_dimension"] = 32
            raw[module]["attention_layer_config"]["dim_feedforward"] = 64
            raw[module]["attention_layer_config"]["num_heads"] = 4
        raw["global_aggregator_config"]["aggregator_config"]["num_heads"] = 4
        raw["global_aggregator_config"]["output_dim"] = 16
        raw["positional_encoding_config"]["N_radial_basis_functions"] = 8
        raw["positional_encoding_config"]["d_projection"] = 8
    config = TypeAdapter(ArchitectureConfig).validate_python(raw)
    assert isinstance(config, EncoderDecoderArchitectureConfig)
    return config


@dataclass
class EncoderDecoder:
    encoder: nn.Module
    decoder: nn.Module
    dimensions: ModelDimensions


def _build(config: EncoderDecoderArchitectureConfig, device: str) -> EncoderDecoder:
    torch.manual_seed(0)
    dimensions = config.dimensions
    aggregator = config.global_aggregator_config.build(dimensions)
    encoder = config.encoder_config.build(aggregator, dimensions)
    decoder = config.decoder_config.build(dimensions)
    # The zero-initialised projections (pair bias, pair transition output,
    # decoder head) would hide errors in everything behind them.
    for module in (encoder, decoder):
        for parameter in module.parameters():
            if not parameter.abs().gt(0).any():
                nn.init.normal_(parameter, std=0.1)
    return EncoderDecoder(
        encoder=encoder.float().to(device).train(),
        decoder=decoder.float().to(device).train(),
        dimensions=dimensions,
    )


@dataclass
class Batch:
    sample: PreprocessedSample
    noise_level: torch.Tensor
    target: torch.Tensor


def _batch(
    dimensions: ModelDimensions, molecules: int, atoms: int, device: str
) -> Batch:
    """Random inputs with the preprocessor's shapes; molecule 0 fills the
    padding, the others are shorter."""
    generator = torch.Generator().manual_seed(molecules * 1000 + atoms)
    lengths = torch.randint(atoms // 2, atoms + 1, (molecules,), generator=generator)
    lengths[0] = atoms
    padding_mask = torch.arange(atoms)[None, :] >= lengths[:, None]
    pair_mask = ~(padding_mask[:, :, None] | padding_mask[:, None, :])
    pair = torch.randn(
        molecules, atoms, atoms, dimensions.pair_dimension, generator=generator
    )
    pair = 0.5 * (pair + pair.transpose(1, 2)) * pair_mask[..., None]
    radial_basis = torch.rand(
        molecules,
        atoms,
        atoms,
        dimensions.radial_basis_dimension,
        generator=generator,
    )
    sample = PreprocessedSample(
        preprocessed_atomic_embeddings=torch.randn(
            molecules,
            atoms,
            dimensions.atomic_descriptor_dimension,
            generator=generator,
        ).to(device),
        padding_mask=padding_mask.to(device),
        initial_pair_representation=pair.to(device),
        geometrical_encoding=radial_basis.to(device),
        pair_mask=pair_mask.to(device),
    )
    return Batch(
        sample=sample,
        noise_level=(torch.rand(molecules, generator=generator) + 0.1).to(device),
        target=torch.randn(
            molecules, atoms, dimensions.target_dimension, generator=generator
        ).to(device),
    )


@dataclass
class StepResult:
    descriptor: torch.Tensor
    decoder_output: torch.Tensor
    gradients: dict[str, torch.Tensor]


def _step(modules: EncoderDecoder, batch: Batch) -> StepResult:
    """Encoder and decoder forward, a masked squared error, and backward."""
    for module in (modules.encoder, modules.decoder):
        module.zero_grad(set_to_none=True)
    descriptor: MolecularDescriptor = modules.encoder(batch.sample)
    decoder_output = modules.decoder(batch.sample, descriptor, batch.noise_level)
    assert batch.sample.padding_mask is not None
    real_atoms = (~batch.sample.padding_mask)[..., None]
    loss = ((decoder_output - batch.target).square() * real_atoms).mean()
    loss.backward()
    # A compiled module names its parameters "_orig_mod.<name>".
    gradients = {
        f"{name}.{parameter_name.removeprefix('_orig_mod.')}": (
            parameter.grad.detach().clone()
        )
        for name, module in (("encoder", modules.encoder), ("decoder", modules.decoder))
        for parameter_name, parameter in module.named_parameters()
        if parameter.grad is not None
    }
    return StepResult(
        descriptor=descriptor.tokens.detach(),
        decoder_output=decoder_output.detach(),
        gradients=gradients,
    )


def _compiled(modules: EncoderDecoder, backend: str | Callable = "inductor"):
    return EncoderDecoder(
        encoder=cast(
            nn.Module, torch.compile(modules.encoder, dynamic=True, backend=backend)
        ),
        decoder=cast(
            nn.Module, torch.compile(modules.decoder, dynamic=True, backend=backend)
        ),
        dimensions=modules.dimensions,
    )


# --- the pair linear --------------------------------------------------------


def test_pair_channel_matmul_passes_the_operator_checks():
    generator = torch.Generator().manual_seed(0)
    pair = torch.randn(2, 5, 5, 6, generator=generator, requires_grad=True)
    weight = torch.randn(3, 6, generator=generator, requires_grad=True)
    torch.library.opcheck(pair_channel_matmul, (pair, weight))


def test_pair_channel_matmul_rejects_mismatched_inputs():
    pair = torch.randn(2, 5, 5, 6, device="meta")
    with pytest.raises(ValueError, match="channels"):
        pair_channel_matmul(pair, torch.randn(3, 4, device="meta"))
    with pytest.raises(ValueError, match="the weight"):
        pair_channel_matmul(pair, torch.randn(3, 6, device="meta").double())


def _linear_pair(bias: bool) -> tuple[nn.Linear, PairLinear]:
    torch.manual_seed(0)
    linear = nn.Linear(6, 3, bias=bias)
    pair_linear = PairLinear(6, 3, bias=bias)
    pair_linear.load_state_dict(linear.state_dict())
    return linear, pair_linear


def _values_and_gradients(
    module: nn.Module, pair: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    pair = pair.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    output = module(pair)
    output.float().square().sum().backward()
    gradients = [pair.grad] + [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None for gradient in gradients)
    return output.detach(), [
        gradient.detach().clone() for gradient in gradients if gradient is not None
    ]


def _compile_eager_backend(module: nn.Module) -> nn.Module:
    """Traced by dynamo (so ``is_compiling`` holds and the custom operator runs)
    but executed eagerly: no C++ compiler needed."""
    return cast(nn.Module, torch.compile(module, dynamic=True, backend="aot_eager"))


@pytest.mark.parametrize("bias", [True, False])
def test_compiled_pair_linear_matches_linear_in_values_and_gradients(bias: bool):
    linear, pair_linear = _linear_pair(bias)
    pair = torch.randn(2, 5, 5, 6)
    expected, expected_gradients = _values_and_gradients(linear, pair)
    output, gradients = _values_and_gradients(_compile_eager_backend(pair_linear), pair)
    torch.testing.assert_close(output, expected)
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("compiled", [False, True])
def test_pair_linear_follows_autocast_like_linear(bias: bool, compiled: bool):
    linear, pair_linear = _linear_pair(bias)
    module = _compile_eager_backend(pair_linear) if compiled else pair_linear
    pair = torch.randn(2, 5, 5, 6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, expected_gradients = _values_and_gradients(linear, pair)
        output, gradients = _values_and_gradients(module, pair)
    assert output.dtype == expected.dtype == torch.bfloat16
    # nn.Linear adds the bias inside the matmul and rounds to bfloat16 once; the
    # compiled PairLinear rounds the product, then adds the bias: one extra
    # bfloat16 rounding (relative 2**-8) on values of order one.
    relative, absolute = 1.6e-2, 1e-2
    torch.testing.assert_close(output, expected, rtol=relative, atol=absolute)
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        assert gradient.dtype == expected_gradient.dtype
        torch.testing.assert_close(
            gradient, expected_gradient, rtol=relative, atol=absolute
        )


def test_pair_linear_exports_without_the_custom_operator():
    """torch.export (for AOTInductor) takes the plain linear: a Python custom
    operator cannot run inside an AOTInductor package."""
    linear, pair_linear = _linear_pair(bias=True)
    pair = torch.randn(2, 5, 5, 6)
    atoms = torch.export.Dim("atoms", min=2)
    exported = torch.export.export(
        pair_linear,
        (pair,),
        dynamic_shapes=({0: torch.export.Dim("molecules", min=2), 1: atoms, 2: atoms},),
        strict=False,
    )
    operators = {str(node.target) for node in exported.graph.nodes}
    assert not any("pair_channel_matmul" in operator for operator in operators)
    larger = torch.randn(3, 7, 7, 6)
    torch.testing.assert_close(exported.module()(larger), linear(larger))


# --- the root cause: no floor-divided sizes in the traced graphs -------------


def _floor_divided_sizes(graph_module: torch.fx.GraphModule) -> list[str]:
    found = []
    for node in graph_module.graph.nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and "//" in str(tuple(value.shape)):
            found.append(f"{node.name}: {tuple(value.shape)}")
    return found


def test_traced_graphs_have_no_floor_divided_sizes():
    """Forward and backward graphs (training mode, with dropout) of both
    modules: every size must be a plain product of the batch symbols."""
    found: list[str] = []

    def record(graph_module, example_inputs):
        found.extend(_floor_divided_sizes(graph_module))
        return make_boxed_func(graph_module.forward)

    backend = aot_autograd(fw_compiler=record, bw_compiler=record)
    modules = _build(_architecture(small=True, dropout=0.1), "cpu")
    torch._dynamo.reset()
    compiled = _compiled(modules, backend=backend)
    for molecules, atoms in BATCH_SHAPES[:2]:
        _step(compiled, _batch(modules.dimensions, molecules, atoms, "cpu"))
    assert found == []


# --- compiled against eager --------------------------------------------------


def _assert_compiled_matches_eager(device: str, *, small: bool) -> None:
    eager = _build(_architecture(small=small, dropout=0.0), device)
    compiled = _compiled(
        EncoderDecoder(
            encoder=copy.deepcopy(eager.encoder),
            decoder=copy.deepcopy(eager.decoder),
            dimensions=eager.dimensions,
        )
    )
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit=1, fail_on_recompile_limit_hit=True
    ):
        for molecules, atoms in BATCH_SHAPES:
            batch = _batch(eager.dimensions, molecules, atoms, device)
            expected = _step(eager, batch)
            result = _step(compiled, batch)
            torch.testing.assert_close(
                result.descriptor,
                expected.descriptor,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
            torch.testing.assert_close(
                result.decoder_output,
                expected.decoder_output,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
            assert result.gradients.keys() == expected.gradients.keys()
            for name, gradient in expected.gradients.items():
                torch.testing.assert_close(
                    result.gradients[name],
                    gradient,
                    rtol=RELATIVE_TOLERANCE,
                    atol=ABSOLUTE_TOLERANCE,
                    msg=lambda message, name=name: f"{name}: {message}",
                )


def _assert_training_with_dropout_is_one_graph(device: str, *, small: bool) -> None:
    modules = _compiled(_build(_architecture(small=small, dropout=0.1), device))
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit=1, fail_on_recompile_limit_hit=True
    ):
        for molecules, atoms in BATCH_SHAPES:
            result = _step(
                modules, _batch(modules.dimensions, molecules, atoms, device)
            )
            assert torch.isfinite(result.decoder_output).all()
            assert all(
                torch.isfinite(gradient).all() for gradient in result.gradients.values()
            )


@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_matches_eager_on_cpu():
    _assert_compiled_matches_eager("cpu", small=True)


@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_training_with_dropout_is_one_graph_on_cpu():
    _assert_training_with_dropout_is_one_graph("cpu", small=True)


@needs_cuda
@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_matches_eager_on_cuda():
    """The drug_like_small widths: the CantSplit failure was in the triton
    code generation, which only CUDA runs."""
    _assert_compiled_matches_eager("cuda", small=False)


@needs_cuda
@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_training_with_dropout_is_one_graph_on_cuda():
    _assert_training_with_dropout_is_one_graph("cuda", small=False)
