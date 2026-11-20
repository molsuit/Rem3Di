"""A linear map over the channels of a pair tensor ``(molecules, atoms, atoms,
channels)`` that torch.compile traces with clean dynamic sizes.

Why not ``nn.Linear``: on a tensor with more than two dimensions the linear
flattens the leading dimensions into one, multiplies, and views the result
back. With dynamic shapes the flattened size of a pair tensor is
``molecules * atoms**2``, and splitting it back into ``(molecules, atoms,
atoms)`` gives the last atom axis the size ``(atoms**2) // atoms``: torch's
symbolic floor division only cancels factors that appear literally in both
operands, so it never reduces ``atoms**2 // atoms`` to ``atoms`` (torch 2.15).
Inductor then fuses tensors whose sizes say ``atoms`` with tensors whose sizes
say ``atoms**2 // atoms``, cannot prove the two equal, and the triton code
generation of the backward fails with ``CantSplit: 64*s not divisible by
((s**2)//s)``. Every view that splits a merged ``atoms**2`` hits this, whatever
the order of the dimensions, so the fix cannot be a different reshape.

Under torch.compile, :class:`PairLinear` calls :func:`pair_channel_linear`, a
custom operator whose output size is declared to be the input size with the
last dimension replaced, and whose backward is built from the same operator and
from flattening reshapes only (which merge dimensions and never split them).
The computation is the plain ``F.linear``; in eager mode the module calls
``F.linear`` directly and skips the custom operator's dispatch overhead.

Equivalent formulations without a custom operator were checked and rejected: a
batched matmul against the broadcast weight traces clean sizes but its backward
materialises one weight gradient per atom; a 1x1 convolution traces clean sizes
but runs in TF32 under cuDNN's default settings; ``as_strided`` instead of the
view brings the split back in its backward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.library.custom_op("remedi::pair_channel_linear", mutates_args=())
def pair_channel_linear(
    pair: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    """``F.linear`` over the last dimension of ``pair`` (any number of leading
    dimensions); the result is contiguous."""
    return F.linear(pair, weight, bias).contiguous()


@pair_channel_linear.register_fake
def _pair_channel_linear_fake(
    pair: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    if pair.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"pair_channel_linear: input has {pair.shape[-1]} channels, the weight "
            f"expects {weight.shape[1]}"
        )
    return pair.new_empty((*pair.shape[:-1], weight.shape[0]))


# torch passes the context, inputs and output by these keyword names.
def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    pair, weight, bias = inputs
    ctx.save_for_backward(pair, weight)
    ctx.has_bias = bias is not None


def _backward(
    ctx: Any, output_gradient: Tensor
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    pair, weight = ctx.saved_tensors
    # Without a bias argument there are only two entries.
    needs_pair, needs_weight, *needs_bias_entry = ctx.needs_input_grad
    needs_bias = bool(needs_bias_entry) and needs_bias_entry[0]
    flat_output_gradient = output_gradient.reshape(-1, output_gradient.shape[-1])

    pair_gradient = (
        pair_channel_linear(output_gradient, weight.t()) if needs_pair else None
    )
    weight_gradient = (
        flat_output_gradient.t() @ pair.reshape(-1, pair.shape[-1])
        if needs_weight
        else None
    )
    bias_gradient = (
        flat_output_gradient.sum(dim=0) if needs_bias and ctx.has_bias else None
    )
    return pair_gradient, weight_gradient, bias_gradient


pair_channel_linear.register_autograd(_backward, setup_context=_setup_context)


class PairLinear(nn.Linear):
    """``nn.Linear`` over the channels of a pair tensor (same parameters, same
    state dict), compilable with dynamic atom counts; see the module docstring."""

    # ``input`` as in ``nn.Linear.forward``, which this overrides.
    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        if torch.compiler.is_compiling():
            return pair_channel_linear(input, self.weight, self.bias)
        return F.linear(input, self.weight, self.bias)
