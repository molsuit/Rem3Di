"""The optimizer's parameter groups and the learning-rate schedules."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from torch import nn

from remedi.configuration.training_config import (
    ConstantScheduleConfig,
    TrainingRunConfig,
    WarmUpCosineScheduleConfig,
)
from remedi.model.pair_linear import PairLinear
from remedi.training.optimization import parameter_groups


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8)
        self.pair_projection = PairLinear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.elements = nn.Embedding(10, 8)
        self.query = nn.Parameter(torch.randn(2, 4))


def _names(module: nn.Module, parameters: list[nn.Parameter]) -> set[str]:
    identities = {id(parameter) for parameter in parameters}
    return {
        name
        for name, parameter in module.named_parameters()
        if id(parameter) in identities
    }


def test_only_linear_weight_matrices_are_decayed() -> None:
    model = _Model()
    groups = parameter_groups([model])
    assert _names(model, groups.decayed) == {
        "projection.weight",
        "pair_projection.weight",
    }
    assert _names(model, groups.undecayed) == {
        "projection.bias",
        "pair_projection.bias",
        "norm.weight",
        "norm.bias",
        "elements.weight",
        "query",
    }


def test_frozen_and_shared_parameters() -> None:
    model = _Model()
    model.projection.requires_grad_(False)
    groups = parameter_groups([model, model.norm])
    assert "projection.weight" not in _names(model, groups.all)
    assert len(groups.all) == len({id(parameter) for parameter in groups.all})


def test_optimizer_groups_carry_the_weight_decay() -> None:
    groups = parameter_groups([_Model()]).for_optimizer(0.05)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.05, 0.0]


def _learning_rates(schedule, steps: int, peak: float = 1.0) -> list[float]:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=peak)
    scheduler = schedule.build(optimizer, total_steps=steps)
    rates = []
    for _ in range(steps):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return rates


def test_warm_up_cosine_shape() -> None:
    rates = _learning_rates(
        WarmUpCosineScheduleConfig(
            warm_up_fraction=0.1, final_learning_rate_fraction=0.01
        ),
        steps=100,
    )
    # Linear warm-up over 10 steps, starting above zero.
    assert rates[:10] == pytest.approx([(step + 1) / 10 for step in range(10)])
    assert rates[10] == pytest.approx(1.0)
    # Monotone cosine decay towards 1 % of the peak at the last step.
    assert all(later <= earlier for earlier, later in itertools.pairwise(rates[10:]))
    halfway = 10 + 45
    assert rates[halfway] == pytest.approx(0.01 + 0.99 * 0.5, abs=1e-9)
    assert rates[-1] == pytest.approx(
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * 89 / 90))
    )


def test_constant_schedule() -> None:
    rates = _learning_rates(ConstantScheduleConfig(warm_up_fraction=0.04), steps=50)
    assert rates[0] == pytest.approx(0.5)
    assert rates[1:] == pytest.approx([1.0] * 49)
    assert _learning_rates(ConstantScheduleConfig(), steps=5) == [1.0] * 5


def test_schedule_state_records_its_plan() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    first = WarmUpCosineScheduleConfig().build(
        torch.optim.SGD([parameter], lr=1.0), total_steps=100
    )
    second = WarmUpCosineScheduleConfig().build(
        torch.optim.SGD([parameter], lr=1.0), total_steps=40
    )
    second.load_state_dict(first.state_dict())
    assert second.total_steps == 100


def test_the_schedule_is_selected_by_kind() -> None:
    values = {
        "training_name": "run",
        "dataloader": {
            "batch_sampling": {"kind": "bucket_batch", "max_atoms_per_batch": 10}
        },
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "model_config_path": "architecture.yaml",
        "output_base": "runs",
    }
    assert isinstance(
        TrainingRunConfig.model_validate(values).learning_rate_schedule,
        WarmUpCosineScheduleConfig,
    )
    constant = TrainingRunConfig.model_validate(
        {**values, "learning_rate_schedule": {"kind": "constant"}}
    )
    assert isinstance(constant.learning_rate_schedule, ConstantScheduleConfig)
