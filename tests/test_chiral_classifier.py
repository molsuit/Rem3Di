"""Unit tests for the from-scratch chiral classifier building blocks.

Covers the parts that don't need MACE / a GPU:
  * the unified head: ``n_classes`` turns a RegressionHead into a logits
    classifier (right shape, softmax inference, no label-scaling buffers);
  * FocalLoss reduces to cross-entropy at gamma=0 and down-weights easy
    examples at gamma>0; inverse-frequency alpha up-weights rare classes;
  * the supervised collate stacks per-molecule labels into ``(B,)``;
  * make_supervised_getitem attaches the label to a Sample;
  * the chiral architecture yaml validates as a single-head classifier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydantic_yaml as pyaml
import pytest
import torch

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    RegressionArchitectureConfig,
    RegressionHeadConfig,
)
from threedscriptors.data_handling.sample import (
    Sample,
    yield_molecules_supervised_collate_fn,
)
from threedscriptors.model.regression_models import RegressionHead
from threedscriptors.training.classification_training import (
    FocalLoss,
    inverse_frequency_alpha,
)


def test_head_classification_shapes_and_no_scaling() -> None:
    cfg = RegressionHeadConfig(
        task_name="chirality_type", n_classes=5, input_dimensions=16
    )
    head = RegressionHead(cfg)
    assert head.is_classification
    # No regression label-scaling buffers on a classification head.
    assert not hasattr(head, "task_mean")
    x = torch.randn(8, 16)
    logits = head(x)
    assert logits.shape == (8, 5)
    probs = head.inference(x)
    assert probs.shape == (8, 5)
    torch.testing.assert_close(probs.sum(-1), torch.ones(8), atol=1e-5, rtol=1e-5)


def test_head_regression_still_scalar() -> None:
    # n_classes None -> scalar regression head; task_config carries mean/std.
    import types

    cfg = RegressionHeadConfig(task_name="y", input_dimensions=16)
    cfg.task_config = types.SimpleNamespace(mean=0.0, std=1.0, scaling=None)
    head = RegressionHead(cfg)
    assert not head.is_classification
    assert head(torch.randn(4, 16)).shape == (4, 1)
    assert hasattr(head, "task_mean")


def test_focal_loss_equals_cross_entropy_at_gamma_zero() -> None:
    logits = torch.randn(32, 5, generator=torch.Generator().manual_seed(0))
    target = torch.randint(0, 5, (32,), generator=torch.Generator().manual_seed(1))
    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(fl(logits, target), ce, atol=1e-6, rtol=1e-6)


def test_focal_loss_downweights_easy_examples() -> None:
    # One confident-correct (easy) example: focal loss < cross-entropy.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    ce = torch.nn.functional.cross_entropy(logits, target)
    fl = FocalLoss(gamma=2.0)(logits, target)
    assert fl < ce


def test_inverse_frequency_alpha_upweights_rare() -> None:
    labels = np.array([0] * 100 + [1] * 10 + [2] * 1)
    alpha = inverse_frequency_alpha(labels, n_classes=5)
    assert alpha.shape == (5,)
    # Rarer class -> larger weight (preserved through mean-1 normalization).
    assert alpha[2] > alpha[1] > alpha[0]
    # Weights are normalized to mean 1 so the loss scale tracks cross-entropy.
    assert float(alpha.mean()) == pytest.approx(1.0, abs=1e-5)
    # Absent classes (3, 4) share the same neutral weight.
    assert float(alpha[3]) == pytest.approx(float(alpha[4]), abs=1e-6)


def test_supervised_collate_stacks_labels() -> None:
    def _mk(label: int) -> Sample:
        return Sample(
            atomic_positions=torch.zeros(3, 3),
            atomic_numbers=torch.ones(3, dtype=torch.long),
            total_charge=torch.tensor(0.0),
            multiplicity=torch.tensor(1.0),
            regression_targets=torch.tensor(label, dtype=torch.long),
        )

    batch = yield_molecules_supervised_collate_fn([_mk(0), _mk(3), _mk(1)])
    assert batch.regression_targets.shape == (3,)
    assert batch.regression_targets.tolist() == [0, 3, 1]
    # 3 molecules x 3 atoms concatenated flat for on-the-fly MACE.
    assert batch.atomic_positions.shape == (9, 3)
    assert batch.system_index.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_chiral_architecture_yaml_is_single_head_classifier() -> None:
    path = Path("configs/training/chiral_cat/architecture_config.yaml")
    ac = pyaml.parse_yaml_file_as(ArchitectureConfig, path)
    assert isinstance(ac, RegressionArchitectureConfig)
    assert ac.embedding_preprocess_config.pseudoscalars is True
    assert ac.embedding_preprocess_config.output_irreps_dim == 384
    heads = ac.regression_head_config
    assert len(heads) == 1
    assert heads[0].n_classes == 5
    # Head input dim cascades from the aggregator's flat descriptor dim.
    assert heads[0].input_dimensions == ac.global_aggregator_config.descriptor_flat_dim
