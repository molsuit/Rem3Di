"""Regression / classification heads built from RegressionHeadConfig."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from remedi.configuration.architecture_config import HeadType, RegressionHeadConfig
from remedi.model.regression_models import (
    FullyConnectedBlock,
    MultitaskHeads,
    RegressionHead,
)


def test_linear_head_maps_the_input_to_the_output():
    # Regression: head_type linear built Linear(hidden_dimensions[-1], ...) and
    # crashed on the input width.
    config = RegressionHeadConfig(
        task_name="y", head_type=HeadType.LINEAR, hidden_dimensions=[]
    )
    head = RegressionHead(config, input_dimension=12)
    assert head(torch.randn(3, 12)).shape == (3, 1)
    linear_layers = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 1
    assert linear_layers[0].in_features == 12


def test_linear_head_rejects_hidden_dimensions():
    with pytest.raises(ValidationError, match="hidden_dimensions"):
        RegressionHeadConfig(
            task_name="y", head_type=HeadType.LINEAR, hidden_dimensions=[64]
        )


@pytest.mark.parametrize(
    "activation, activation_class",
    [("silu", torch.nn.SiLU), ("gelu", torch.nn.GELU), ("relu", torch.nn.ReLU)],
)
def test_fully_connected_head_uses_configured_activation_and_dropout(
    activation, activation_class
):
    # Regression: FullyConnectedBlock hard-coded SiLU and Dropout(0.1).
    config = RegressionHeadConfig(
        task_name="y",
        activation=activation,
        dropout=0.25,
        hidden_dimensions=[16, 8],
    )
    head = RegressionHead(config, input_dimension=12)
    blocks = [m for m in head.modules() if isinstance(m, FullyConnectedBlock)]
    assert len(blocks) == 2
    for block in blocks:
        assert isinstance(block.block.activation, activation_class)
        assert block.block.dropout.p == pytest.approx(0.25)
    # Each block owns its activation instance.
    assert blocks[0].block.activation is not blocks[1].block.activation


def test_residual_head_runs():
    config = RegressionHeadConfig(
        task_name="y", head_type=HeadType.RESIDUAL, hidden_dimensions=[16, 16]
    )
    head = RegressionHead(config, input_dimension=12)
    assert head(torch.randn(3, 12)).shape == (3, 1)


def test_head_input_width_includes_auxiliary_data():
    configs = [
        RegressionHeadConfig(task_name="plain", hidden_dimensions=[8]),
        RegressionHeadConfig(
            task_name="with_auxiliary",
            hidden_dimensions=[8],
            auxiliary_input_dimension=3,
        ),
    ]
    heads = MultitaskHeads(configs, descriptor_flat_dimension=10)
    first_norm = {
        name: next(m for m in head.modules() if isinstance(m, torch.nn.LayerNorm))
        for name, head in heads.task_heads.items()
    }
    assert first_norm["plain"].normalized_shape == (10,)
    assert first_norm["with_auxiliary"].normalized_shape == (13,)
