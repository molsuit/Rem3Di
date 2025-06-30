import pytest
import json
from enum import Enum
from pydantic import ValidationError, TypeAdapter
from typing import Union

# import your actual classes here!
from threedscriptors.model.pooling import MeanPool, AttnPool
from threedscriptors.configuration.architecture_config import (
    Aggregations,
    MeanAggregatorConfig,
    AttentionAggregatorConfig,
    GlobalAggregatorConfig
)




@pytest.mark.parametrize(
    "raw, expected",
    [
        ("mean", Aggregations.MEAN),
        (" attention ", Aggregations.ATTENTION),
        (MeanPool, Aggregations.MEAN),
        (AttnPool(d_in=8), Aggregations.ATTENTION),
    ],
)
def test_enum_coercion(raw, expected):
    assert Aggregations(raw) is expected


def test_enum_str():
    assert str(Aggregations.ATTENTION) == "attention"


# ---------------------------------------------------------------------------
# ATTNPOOL assertion check
# ---------------------------------------------------------------------------

def test_attnpool_bad_divisibility():
    with pytest.raises(AssertionError):
        AttnPool(d_in=10, n_heads=4)      # 10 % 4 ⇒ assertion


# ---------------------------------------------------------------------------
# CHILD CONFIGS (Literal enforced)
# ---------------------------------------------------------------------------

def test_mean_cfg_roundtrip():
    cfg = MeanAggregatorConfig(aggregator_type=Aggregations.MEAN)
    assert cfg.aggregator_type is Aggregations.MEAN
    assert '"aggregator_type":"mean"' in cfg.model_dump_json()


def test_mean_cfg_rejects_string():
    with pytest.raises(ValidationError):
        MeanAggregatorConfig(aggregator_type="mean")   # must be enum member


def test_attention_cfg_success():
    cfg = AttentionAggregatorConfig(
        aggregator_type=Aggregations.ATTENTION,
        num_heads=8,
        head_dim=16,
    )
    assert cfg.num_heads == 8
    assert '"aggregator_type":"attention"' in cfg.model_dump_json()


def test_attention_cfg_missing_fields():
    with pytest.raises(ValidationError):
        AttentionAggregatorConfig(aggregator_type=Aggregations.ATTENTION)


# ---------------------------------------------------------------------------
# GLOBAL CONFIG (nested union via discriminator)
# ---------------------------------------------------------------------------

adapter = TypeAdapter(GlobalAggregatorConfig)   # re-usable helper

@pytest.mark.parametrize(
    "payload, expected_cls",
    [
        (
            {
                "aggregator_type_config": {"aggregator_type": Aggregations.MEAN},
                "input_dim": 32,
                "output_dim": 16,
            },
            MeanAggregatorConfig,
        ),
        (
            {
                "aggregator_type_config": {
                    "aggregator_type": Aggregations.ATTENTION,
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
    cfg = adapter.validate_python(payload)          # ← v2 way
    assert isinstance(cfg.aggregator_type_config, expected_cls)


def test_global_invalid_discriminant():
    bad = {
        "aggregator_type_config": {"aggregator_type": "foo"},
        "input_dim": 1,
        "output_dim": 1,
    }
    with pytest.raises(ValidationError):
        adapter.validate_python(bad)