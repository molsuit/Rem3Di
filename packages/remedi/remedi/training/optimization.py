"""The optimizer's parameter groups and the learning-rate schedule of a training run."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class ParameterGroups:
    """The trained parameters, split by whether AdamW decays them."""

    decayed: list[nn.Parameter]
    undecayed: list[nn.Parameter]

    @property
    def all(self) -> list[nn.Parameter]:
        return [*self.decayed, *self.undecayed]

    def for_optimizer(self, weight_decay: float) -> list[dict[str, Any]]:
        return [
            {"params": self.decayed, "weight_decay": weight_decay},
            {"params": self.undecayed, "weight_decay": 0.0},
        ]


def parameter_groups(modules: Iterable[nn.Module]) -> ParameterGroups:
    """Split the trained parameters of ``modules``: weight decay on the weight
    matrices of linear layers only.

    Biases, normalisation scales, embedding tables, pooling queries and seeds stay
    undecayed: shrinking them towards zero is not a regularisation of the mapping
    (a LayerNorm scale of zero switches its channel off). Frozen parameters (MACE)
    are left out; a parameter shared between modules appears once.
    """
    modules = list(modules)
    decayed_identities = {
        id(submodule.weight)
        for module in modules
        for submodule in module.modules()
        if isinstance(submodule, nn.Linear)
    }
    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if id(parameter) in decayed_identities:
                decayed.append(parameter)
            else:
                undecayed.append(parameter)
    return ParameterGroups(decayed=decayed, undecayed=undecayed)


class WarmUpThenDecay:
    """The learning-rate factor at a step: a linear warm-up over ``warm_up_steps``,
    then either constant or a cosine decay to ``final_fraction`` at ``total_steps``.

    A plain object rather than a closure, so ``LambdaLR`` saves its attributes with
    the scheduler state and a resumed run can tell which plan it belongs to.
    """

    def __init__(
        self,
        *,
        total_steps: int,
        warm_up_steps: int,
        cosine_decay: bool,
        final_fraction: float,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.total_steps = int(total_steps)
        self.warm_up_steps = int(warm_up_steps)
        self.cosine_decay = bool(cosine_decay)
        self.final_fraction = float(final_fraction)

    def __call__(self, step: int) -> float:
        if step < self.warm_up_steps:
            # The first step already trains (factor 1/warm_up_steps, not 0).
            return (step + 1) / self.warm_up_steps
        if not self.cosine_decay:
            return 1.0
        decay_steps = max(self.total_steps - self.warm_up_steps, 1)
        progress = min((step - self.warm_up_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_fraction + (1.0 - self.final_fraction) * cosine


class PlannedLearningRateSchedule(LambdaLR):
    """``LambdaLR`` over one ``WarmUpThenDecay`` for all parameter groups, with the
    number of steps it was planned for (restored from a saved state)."""

    def __init__(self, optimizer: torch.optim.Optimizer, factor: WarmUpThenDecay):
        super().__init__(optimizer, lr_lambda=factor)

    @property
    def total_steps(self) -> int:
        factor = self.lr_lambdas[0]
        assert isinstance(factor, WarmUpThenDecay)
        return factor.total_steps
