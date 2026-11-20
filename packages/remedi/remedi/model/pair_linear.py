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

Under torch.compile, :class:`PairLinear` multiplies by the weight through
:func:`pair_channel_matmul`, a custom operator whose output size is declared to
be the input size with the last dimension replaced, and whose backward is built
from the same operator and from flattening reshapes only (which merge dimensions
and never split them). The bias is added outside the operator, as a plain
broadcast add, so inductor can fuse it with what follows (an activation, a
residual). In eager mode the module calls ``F.linear`` directly.

Autocast: the operator is opaque to autocast, so :class:`PairLinear` casts the
input and the parameters to the autocast dtype itself when autocast is active on
the input's device, as autocast does for ``F.linear``.

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


@torch.library.custom_op("remedi::pair_channel_matmul", mutates_args=())
def pair_channel_matmul(pair: Tensor, weight: Tensor) -> Tensor:
    """``pair @ weight.T`` over the last dimension of ``pair`` (any number of
    leading dimensions); the result is contiguous."""
    return F.linear(pair, weight).contiguous()


@pair_channel_matmul.register_fake
def _pair_channel_matmul_fake(pair: Tensor, weight: Tensor) -> Tensor:
    if pair.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"pair_channel_matmul: input has {pair.shape[-1]} channels, the weight "
            f"expects {weight.shape[1]}"
        )
    if pair.dtype != weight.dtype:
        raise ValueError(
            f"pair_channel_matmul: input is {pair.dtype}, the weight {weight.dtype}"
        )
    return pair.new_empty((*pair.shape[:-1], weight.shape[0]))


# torch passes the context, inputs and output by these keyword names.
def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    pair, weight = inputs
    ctx.save_for_backward(pair, weight)


def _backward(ctx: Any, output_gradient: Tensor) -> tuple[Tensor | None, Tensor | None]:
    pair, weight = ctx.saved_tensors
    needs_pair, needs_weight = ctx.needs_input_grad
    pair_gradient = (
        pair_channel_matmul(output_gradient, weight.t().contiguous())
        if needs_pair
        else None
    )
    weight_gradient = (
        output_gradient.reshape(-1, output_gradient.shape[-1]).t()
        @ pair.reshape(-1, pair.shape[-1])
        if needs_weight
        else None
    )
    return pair_gradient, weight_gradient


pair_channel_matmul.register_autograd(_backward, setup_context=_setup_context)


def _autocast_dtype(pair: Tensor) -> torch.dtype | None:
    """The autocast dtype on ``pair``'s device, or None when autocast is off."""
    device_type = pair.device.type
    if not torch.is_autocast_enabled(device_type):
        return None
    return torch.get_autocast_dtype(device_type)


class PairLinear(nn.Linear):
    """``nn.Linear`` over the channels of a pair tensor (same parameters, same
    state dict), compilable with dynamic atom counts; see the module docstring."""

    # ``input`` as in ``nn.Linear.forward``, which this overrides.
    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        # torch.export (AOTInductor) traces inference only, where nn.Linear's view
        # compiles; and a Python custom operator cannot run in an AOTInductor
        # package, so the export takes the plain linear.
        if not torch.compiler.is_compiling() or torch.compiler.is_exporting():
            return F.linear(input, self.weight, self.bias)
        pair, weight, bias = input, self.weight, self.bias
        autocast_dtype = _autocast_dtype(pair)
        if autocast_dtype is not None:
            pair = pair.to(autocast_dtype)
            weight = weight.to(autocast_dtype)
            bias = None if bias is None else bias.to(autocast_dtype)
        output = pair_channel_matmul(pair, weight)
        return output if bias is None else output + bias
