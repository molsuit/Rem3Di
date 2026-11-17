"""Shared pieces of the attentions built on ``scaled_dot_product_attention``."""

from __future__ import annotations

import torch
from torch import Tensor


def key_padding_bias(padding_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """An additive ``(B, 1, 1, N)`` attention mask from a ``(B, N)`` padding mask.

    ``padding_mask`` is True for padded atoms. Their keys get the dtype's lowest
    finite value rather than ``-inf``, so a query whose keys are all padded
    still yields finite (if meaningless) weights instead of NaN.
    """
    bias = torch.zeros(padding_mask.shape, dtype=dtype, device=padding_mask.device)
    return bias.masked_fill(padding_mask.bool(), torch.finfo(dtype).min)[
        :, None, None, :
    ]
