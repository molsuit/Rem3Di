"""Tests for benchmark coverage accounting."""

from __future__ import annotations

import pytest

from remedi.evaluation.benchmark.coverage import (
    DEFAULT_COVERAGE_THRESHOLD,
    comparable,
    coverage_for,
)


def test_full_coverage_is_not_limited():
    cov = coverage_for(n_evaluated=1000, n_expected=1000)
    assert cov.fraction == 1.0
    assert cov.n_dropped == 0
    assert not cov.is_limited


def test_the_off24_vdss_case():
    """The motivating example: OFF24's ten elements reach 43.4% of TDC VDss."""
    cov = coverage_for(n_evaluated=434, n_expected=1000)
    assert cov.fraction == pytest.approx(0.434)
    assert cov.is_limited
    assert "COVERAGE-LIMITED" in cov.describe()


def test_threshold_boundary_is_inclusive_above():
    assert not coverage_for(n_evaluated=90, n_expected=100).is_limited
    assert coverage_for(n_evaluated=89, n_expected=100).is_limited
    assert DEFAULT_COVERAGE_THRESHOLD == 0.90


def test_evaluating_more_than_expected_is_rejected():
    """Guards the likeliest mistake: counting n_expected after filtering."""
    with pytest.raises(ValueError, match="after filtering"):
        coverage_for(n_evaluated=100, n_expected=90)


def test_empty_expected_set_is_zero_not_one():
    assert coverage_for(n_evaluated=0, n_expected=0).fraction == 0.0


def test_negative_counts_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        coverage_for(n_evaluated=-1, n_expected=10)


def test_threshold_must_be_a_fraction():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        coverage_for(n_evaluated=1, n_expected=2, threshold=1.5)


def test_two_limited_results_are_still_not_comparable():
    """Equally limited is not the same as comparable — they drop different molecules."""
    a = coverage_for(n_evaluated=434, n_expected=1000)
    b = coverage_for(n_evaluated=434, n_expected=1000)
    assert not comparable(a, b)


def test_comparable_requires_both_sides_full():
    full = coverage_for(n_evaluated=990, n_expected=1000)
    thin = coverage_for(n_evaluated=614, n_expected=1000)
    assert comparable(full, full)
    assert not comparable(full, thin)
    assert not comparable(thin, full)
