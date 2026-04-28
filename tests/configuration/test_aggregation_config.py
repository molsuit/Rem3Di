import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from threedscriptors.configuration.architecture_config import (
    AttentionAggregatorConfig,
    GlobalAggregatorConfig,
    MeanAggregatorConfig,
)
from threedscriptors.model.pooling import AttnPool


def test_attnpool_bad_divisibility():
    with pytest.raises(AssertionError):
        AttnPool(d_in=10, n_heads=4)  # 10 % 4 => assertion


def test_attnpool_projects_to_d_out_when_different():
    pool = AttnPool(d_in=1024, d_out=320, n_heads=8, d_hidden=64)
    x = torch.randn(2, 5, 1024)
    pad_mask = torch.zeros(2, 5, dtype=torch.bool)
    out = pool(x, pad_mask)
    assert out.shape == (2, 320)


def test_attnpool_identity_when_d_in_equals_d_out():
    pool = AttnPool(d_in=128, d_out=128, n_heads=8, d_hidden=64)
    assert isinstance(pool.out_proj, torch.nn.Identity)


def test_mean_cfg_roundtrip():
    cfg = MeanAggregatorConfig()
    assert cfg.aggregator_type == "mean"
    assert '"aggregator_type":"mean"' in cfg.model_dump_json()


def test_attention_cfg_success():
    cfg = AttentionAggregatorConfig(num_heads=8, head_dim=16)
    assert cfg.num_heads == 8
    assert '"aggregator_type":"attention"' in cfg.model_dump_json()


def test_attention_cfg_missing_fields():
    with pytest.raises(ValidationError):
        AttentionAggregatorConfig()


adapter = TypeAdapter(GlobalAggregatorConfig)


@pytest.mark.parametrize(
    "payload, expected_cls",
    [
        (
            {
                "aggregator_type_config": {"aggregator_type": "mean"},
                "input_dim": 32,
                "output_dim": 16,
            },
            MeanAggregatorConfig,
        ),
        (
            {
                "aggregator_type_config": {
                    "aggregator_type": "attention",
                    "num_heads": 4,
                    "head_dim": 16,
                },
                "input_dim": 64,
                "output_dim": 16,
            },
            AttentionAggregatorConfig,
        ),
    ],
)
def test_global_dispatch(payload, expected_cls):
    cfg = adapter.validate_python(payload)
    assert isinstance(cfg.aggregator_type_config, expected_cls)


def test_global_invalid_discriminant():
    bad = {
        "aggregator_type_config": {"aggregator_type": "foo"},
        "input_dim": 1,
        "output_dim": 1,
    }
    with pytest.raises(ValidationError):
        adapter.validate_python(bad)
