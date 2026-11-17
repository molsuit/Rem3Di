"""Estimating the invariant normalisation statistics from training batches."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
    OnTheFlyInvariantNormalization,
    PrecomputedInvariantNormalization,
)
from remedi.model.preprocessing.invariant_statistics import (
    ensure_invariant_statistics,
    estimate_invariant_statistics,
)
from remedi.training.data.sample import Sample

DIMENSION = 3


class _AtomicPreprocessor(nn.Module):
    def __init__(self, normalization: nn.Module) -> None:
        super().__init__()
        self.invariant_normalization = normalization


class _Preprocessor(nn.Module):
    """Stands in for the MACE preprocessor: the raw invariants are the positions."""

    def __init__(self, normalization: nn.Module | None = None) -> None:
        super().__init__()
        self.atomic_preprocessor = _AtomicPreprocessor(
            normalization or PrecomputedInvariantNormalization(DIMENSION)
        )
        self.batches_seen = 0

    def raw_invariants(self, sample: Sample) -> torch.Tensor:
        self.batches_seen += 1
        assert sample.atomic_positions is not None
        return sample.atomic_positions


def _batches(count: int = 4, atoms: int = 500) -> list[Sample]:
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([1.0, 10.0, 0.0])  # the last channel is constant
    shift = torch.tensor([5.0, -2.0, 3.0])
    return [
        Sample(
            atomic_positions=torch.randn(atoms, DIMENSION, generator=generator) * scale
            + shift
        )
        for _ in range(count)
    ]


def test_statistics_match_the_pooled_atoms() -> None:
    batches = _batches()
    statistics = estimate_invariant_statistics(_Preprocessor(), batches, "cpu")
    atoms = torch.cat([batch.atomic_positions for batch in batches]).double()

    assert statistics.atom_count == 2000
    torch.testing.assert_close(statistics.mean, atoms.mean(dim=0))
    torch.testing.assert_close(
        statistics.standard_deviation[:2], atoms.std(dim=0, unbiased=False)[:2]
    )
    # A constant channel is centred, not divided by ~0.
    assert statistics.constant_channel_count == 1
    assert statistics.standard_deviation[2] == 1.0


def test_ensure_sets_the_statistics_once_from_the_first_batches() -> None:
    preprocessor = _Preprocessor()
    normalization = preprocessor.atomic_preprocessor.invariant_normalization
    statistics = ensure_invariant_statistics(
        preprocessor, _batches(), max_batches=2, device="cpu"
    )
    assert statistics is not None and statistics.atom_count == 1000
    assert preprocessor.batches_seen == 2
    assert bool(normalization.stats_set)

    normalised = normalization(_batches()[0].atomic_positions[None], None)[0]
    assert normalised[:, :2].mean().abs() < 0.2
    assert normalised[:, :2].std().item() == pytest.approx(1.0, abs=0.1)

    # Already set (as after loading a checkpoint): nothing is run or changed.
    assert (
        ensure_invariant_statistics(
            preprocessor, _batches(), max_batches=2, device="cpu"
        )
        is None
    )
    assert preprocessor.batches_seen == 2


def test_loaded_statistics_count_as_set() -> None:
    """Regression: the "set" flag was not saved, so a loaded checkpoint looked
    like one without statistics."""
    preprocessor = _Preprocessor()
    ensure_invariant_statistics(preprocessor, _batches(), max_batches=1, device="cpu")
    reloaded = _Preprocessor()
    reloaded.load_state_dict(preprocessor.state_dict())
    assert bool(reloaded.atomic_preprocessor.invariant_normalization.stats_set)
    assert (
        ensure_invariant_statistics(reloaded, _batches(), max_batches=1, device="cpu")
        is None
    )


def test_on_the_fly_normalisation_is_left_alone() -> None:
    preprocessor = _Preprocessor(OnTheFlyInvariantNormalization(DIMENSION))
    assert (
        ensure_invariant_statistics(
            preprocessor, _batches(), max_batches=1, device="cpu"
        )
        is None
    )
    assert preprocessor.batches_seen == 0


def test_statistics_need_atoms_and_finite_values() -> None:
    with pytest.raises(ValueError, match="at least two atoms"):
        estimate_invariant_statistics(_Preprocessor(), [], "cpu")
    broken = [Sample(atomic_positions=torch.full((4, DIMENSION), float("nan")))]
    with pytest.raises(ValueError, match="not finite"):
        estimate_invariant_statistics(_Preprocessor(), broken, "cpu")
