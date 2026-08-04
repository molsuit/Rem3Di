"""Low-rank adaptation (LoRA) for supervised fine-tuning of a Rem3Di encoder.

Fine-tuning the whole encoder is the strongest adaptation but costs a full
optimiser state per downstream task. LoRA freezes the pretrained weights and
learns a rank-``r`` residual beside each targeted projection, so a task is
carried by a small fraction of the parameters.

On a head-matched 13-task ADMET ladder, relative score across six conditions:

===========================  =====
random init, frozen          0.018
random init + LoRA           0.447
random init + full fine-tune 0.598
pretrained, frozen           0.655
**pretrained + LoRA**        0.814
pretrained + full fine-tune  0.958
===========================  =====

So LoRA recovers most of the gap between a frozen descriptor and full
fine-tuning. It is sensitive to learning rate: ``r=16`` at ``5e-5`` is the
working recipe, while ``1e-4`` collapsed to chance in an earlier sweep.

Usage
-----
.. code-block:: python

    from remedi.training.lora import (
        DEFAULT_LORA_TARGETS, inject_lora_adapters, freeze_non_lora_parameters,
    )

    counts = inject_lora_adapters(model, r=16, alpha=32.0, dropout=0.05)
    freeze_non_lora_parameters(model, also_train=("head",))

The module depends only on ``torch`` — it matches modules by qualified-name
substring, so it works on any ``nn.Module``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

#: Projections adapted by default: per-block attention, plus the PMA pooling
#: projections. Note these are ``.aggregator.*`` and not ``.pool.*`` —
#: ``GlobalAggregatorConfig.build()`` returns the ``PMAAggregator`` directly and
#: ``TransformerPairEncoder`` assigns it as ``self.aggregator``, so there is no
#: intermediate ``pool`` attribute.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    ".attn.W_q",
    ".attn.W_k",
    ".attn.W_v",
    ".attn.W_o",
    ".aggregator.W_Q",
    ".aggregator.W_K",
    ".aggregator.W_V",
)

#: Projections present in only some architecture variants, so their absence is
#: not an error. ``W_O`` is the current :class:`PMAAggregator` output projection;
#: :class:`PMAAggregatorLegacy` has none, so a pre-rewrite checkpoint matches
#: every default target but not this one.
OPTIONAL_LORA_TARGETS: tuple[str, ...] = (".aggregator.W_O",)


class LoraLinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank residual.

    ``B`` is zero-initialised, so at construction the adapter is exactly the
    identity and the wrapped module reproduces the base layer bit for bit. That
    property is what makes it safe to inject into a pretrained model before
    training starts, and it is worth asserting in tests.

    Scaling by ``alpha / r`` keeps the effective learning rate roughly constant
    as the rank changes, so a rank sweep does not double as a learning-rate
    sweep.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.dropout(self.A(x))) * self.scaling


def inject_lora_adapters(
    module: nn.Module,
    *,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_substrings: Sequence[str] = DEFAULT_LORA_TARGETS,
    optional_substrings: Sequence[str] = OPTIONAL_LORA_TARGETS,
    return_per_target: bool = False,
) -> int | dict[str, int]:
    """Wrap every targeted ``nn.Linear`` in a :class:`LoraLinear`, in place.

    Targets are matched case-insensitively as substrings of the qualified module
    name. Raises if **any** target matched nothing, rather than only when the
    total is zero: a partial match is the dangerous case, because it silently
    adapts half the model while still reporting a plausible count.

    ``optional_substrings`` are adapted when present but do not raise when
    absent, for projections that exist in only some architecture variants.
    """
    required = tuple(str(s) for s in target_substrings)
    lowered = {s: s.lower() for s in (*required, *(str(o) for o in optional_substrings))}
    per_target: dict[str, int] = {s: 0 for s in lowered}

    def _recur(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            qual = f"{prefix}.{name}" if prefix else name
            # Match against a dot-prefixed name so a leading "." in a target is
            # segment-anchored at every depth, including the top level: the
            # aggregator is a direct child of the encoder, so its qualified name
            # is "aggregator.W_Q" with no leading dot of its own.
            probe = f".{qual}".lower()
            matched = next(
                (orig for orig, low in lowered.items() if low in probe), None
            )
            if matched is not None and isinstance(child, nn.Linear):
                setattr(
                    parent, name, LoraLinear(child, r=r, alpha=alpha, dropout=dropout)
                )
                per_target[matched] += 1
            else:
                _recur(child, qual)

    _recur(module)

    missing = sorted(t for t in required if per_target[t] == 0)
    if missing:
        raise RuntimeError(
            "LoRA target substrings matched no modules: "
            f"{missing}. Matched: { {k: v for k, v in per_target.items() if v} }. "
            "Module names most likely changed; adapting a subset silently would "
            "leave part of the model frozen."
        )

    return per_target if return_per_target else sum(per_target.values())


def freeze_non_lora_parameters(
    module: nn.Module, *, also_train: Sequence[str] = ()
) -> None:
    """Freeze everything except the LoRA ``A``/``B`` matrices.

    ``also_train`` names further substrings to leave trainable. A task head
    attached to the model is the usual case — without it the head is frozen at
    its initialisation and training appears to run while learning nothing.
    """
    extra = tuple(str(s) for s in also_train)
    for name, param in module.named_parameters():
        param.requires_grad_(
            ".A.weight" in name
            or ".B.weight" in name
            or any(e in name for e in extra)
        )


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
