"""Per-channel statistics of the raw MACE invariants, for their normalisation.

The denoising target is the standardised invariants, so the statistics are
estimated once from training batches before training starts, written into the
preprocessor's ``PrecomputedInvariantNormalization`` and saved with every
checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

import torch

from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
    PrecomputedInvariantNormalization,
)
from remedi.model.preprocessing.preprocessing import PreprocessorWithAtomicEmbedding
from remedi.training.data.sample import Sample

#: Channels whose deviation is below this are constant: they are centred only.
MINIMUM_STANDARD_DEVIATION = 1e-6


@dataclass(frozen=True)
class InvariantStatistics:
    mean: torch.Tensor
    """Per-channel mean, ``(D,)`` float64."""
    standard_deviation: torch.Tensor
    """Per-channel deviation, ``(D,)`` float64; 1 for a constant channel."""
    atom_count: int
    constant_channel_count: int


def estimate_invariant_statistics(
    preprocessor: PreprocessorWithAtomicEmbedding,
    batches: Iterable[Sample],
    device: torch.device | str,
) -> InvariantStatistics:
    """Mean and deviation of every raw invariant channel over the atoms of ``batches``.

    Raises:
        ValueError: if ``batches`` is empty, holds fewer than two atoms, or a
            channel is not finite.
    """
    total: torch.Tensor | None = None
    total_of_squares: torch.Tensor | None = None
    atom_count = 0
    target_device = torch.device(device)
    for batch in batches:
        batch.to_(target_device)
        invariants = preprocessor.raw_invariants(batch).to(torch.float64)
        if total is None or total_of_squares is None:
            total = invariants.new_zeros(invariants.shape[-1])
            total_of_squares = invariants.new_zeros(invariants.shape[-1])
        total += invariants.sum(dim=0)
        total_of_squares += invariants.square().sum(dim=0)
        atom_count += int(invariants.shape[0])
    if total is None or total_of_squares is None or atom_count < 2:
        raise ValueError(
            "Invariant statistics need at least two atoms; "
            f"the batches held {atom_count}."
        )
    mean = total / atom_count
    variance = (total_of_squares / atom_count - mean.square()).clamp_min(0.0)
    standard_deviation = variance.sqrt()
    if not (torch.isfinite(mean).all() and torch.isfinite(standard_deviation).all()):
        raise ValueError("The MACE invariants are not finite; no statistics estimated.")
    constant = standard_deviation < MINIMUM_STANDARD_DEVIATION
    standard_deviation = torch.where(
        constant, torch.ones_like(standard_deviation), standard_deviation
    )
    return InvariantStatistics(
        mean=mean,
        standard_deviation=standard_deviation,
        atom_count=atom_count,
        constant_channel_count=int(constant.sum().item()),
    )


def ensure_invariant_statistics(
    preprocessor: PreprocessorWithAtomicEmbedding,
    batches: Iterable[Sample],
    *,
    max_batches: int,
    device: torch.device | str,
) -> InvariantStatistics | None:
    """Estimate and set the normalisation statistics unless they are in place.

    Returns the estimated statistics, or None when nothing was done: the
    normalisation is not the precomputed kind, or already has its statistics
    (set before, or loaded from a checkpoint).

    Raises:
        ValueError: see :func:`estimate_invariant_statistics`.
    """
    normalization = preprocessor.atomic_preprocessor.invariant_normalization
    if not isinstance(normalization, PrecomputedInvariantNormalization):
        return None
    if bool(normalization.stats_set):
        return None
    statistics = estimate_invariant_statistics(
        preprocessor, islice(batches, max_batches), device
    )
    normalization.set_stats(statistics.mean, statistics.standard_deviation)
    return statistics
