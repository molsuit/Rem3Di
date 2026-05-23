"""Tests for BucketBatchSampler, lengths_from_ptr, and quantize_pad_length."""

from __future__ import annotations

import numpy as np
import pytest

from threedscriptors.training.data.samplers import (
    BucketBatchSampler,
    lengths_from_ptr,
    quantize_pad_length,
)


def _flatten(batches: list[list[int]]) -> list[int]:
    return [i for b in batches for i in b]


def _make_sampler(lengths, **kwargs):
    kwargs.setdefault("generator", np.random.default_rng(0))
    return BucketBatchSampler(lengths, **kwargs)


# ----------------------------- coverage --------------------------------------


def test_every_index_appears_exactly_once():
    rng = np.random.default_rng(0)
    lengths = rng.integers(low=5, high=80, size=200).tolist()
    sampler = _make_sampler(lengths, max_atoms_per_batch=200, shuffle=True)
    flat = _flatten(list(sampler))
    assert sorted(flat) == list(range(len(lengths)))


def test_coverage_no_shuffle():
    sampler = _make_sampler(
        [5, 10, 15, 20, 25], max_atoms_per_batch=30, shuffle=False
    )
    flat = _flatten(list(sampler))
    assert sorted(flat) == [0, 1, 2, 3, 4]


# ----------------------------- token cap -------------------------------------


def test_token_cap_respected():
    sampler = _make_sampler(
        [5, 5, 5, 5, 5, 5, 5, 5], max_atoms_per_batch=15, shuffle=False
    )
    for batch in sampler:
        assert sum(5 for _ in batch) <= 15


def test_token_cap_respected_with_shuffle():
    rng = np.random.default_rng(1)
    lengths = rng.integers(low=2, high=40, size=500).tolist()
    sampler = _make_sampler(
        lengths, max_atoms_per_batch=120, bucket_size=64, shuffle=True
    )
    for batch in sampler:
        if len(batch) == 1 and lengths[batch[0]] > 120:
            continue
        assert sum(lengths[i] for i in batch) <= 120


# ----------------------------- max_batch_size --------------------------------


def test_max_batch_size_respected():
    sampler = _make_sampler(
        [1] * 100, max_atoms_per_batch=10_000, max_batch_size=8, shuffle=False
    )
    for batch in sampler:
        assert len(batch) <= 8


def test_max_batch_size_one():
    sampler = _make_sampler(
        [3, 4, 5], max_atoms_per_batch=100, max_batch_size=1, shuffle=False
    )
    batches = list(sampler)
    assert all(len(b) == 1 for b in batches)
    assert len(batches) == 3


# ----------------------------- oversize handling -----------------------------


def test_oversize_item_yielded_alone():
    lengths = [5, 5, 200, 5, 5]
    with pytest.warns(UserWarning, match="exceed max_atoms_per_batch"):
        sampler = _make_sampler(lengths, max_atoms_per_batch=20, shuffle=False)
    batches = list(sampler)
    oversize_idx = lengths.index(200)
    assert [oversize_idx] in batches


def test_all_items_oversize():
    with pytest.warns(UserWarning):
        sampler = _make_sampler([50, 60, 70], max_atoms_per_batch=10, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 3
    for b in batches:
        assert len(b) == 1


def test_oversize_flushes_partial_batch():
    lengths = [5, 5, 200, 5]
    with pytest.warns(UserWarning):
        sampler = _make_sampler(lengths, max_atoms_per_batch=20, shuffle=False)
    batches = list(sampler)
    assert sorted(_flatten(batches)) == [0, 1, 2, 3]
    assert [2] in batches


# ----------------------------- empty / tiny ----------------------------------


def test_empty_lengths():
    sampler = _make_sampler([], max_atoms_per_batch=10)
    assert list(sampler) == []
    assert len(sampler) == 0


def test_single_item():
    sampler = _make_sampler([7], max_atoms_per_batch=10, shuffle=False)
    assert list(sampler) == [[0]]
    assert len(sampler) == 1


# ----------------------------- bucket_size extremes --------------------------


def test_bucket_size_larger_than_dataset():
    sampler = _make_sampler(
        [3, 4, 5, 6, 7], max_atoms_per_batch=15, bucket_size=10_000, shuffle=True
    )
    flat = _flatten(list(sampler))
    assert sorted(flat) == [0, 1, 2, 3, 4]


def test_bucket_size_one():
    sampler = _make_sampler(
        [3, 4, 5, 6, 7], max_atoms_per_batch=20, bucket_size=1, shuffle=True
    )
    flat = _flatten(list(sampler))
    assert sorted(flat) == [0, 1, 2, 3, 4]


# ----------------------------- input validation ------------------------------


def test_accepts_list_input():
    sampler = _make_sampler([3, 4, 5], max_atoms_per_batch=10, shuffle=False)
    flat = _flatten(list(sampler))
    assert sorted(flat) == [0, 1, 2]


def test_accepts_ndarray_input():
    sampler = _make_sampler(
        np.array([3, 4, 5]), max_atoms_per_batch=10, shuffle=False
    )
    flat = _flatten(list(sampler))
    assert sorted(flat) == [0, 1, 2]


def test_rejects_zero_length():
    with pytest.raises(ValueError, match="positive"):
        BucketBatchSampler([1, 0, 3], max_atoms_per_batch=10)


def test_rejects_negative_length():
    with pytest.raises(ValueError, match="positive"):
        BucketBatchSampler([1, -1, 3], max_atoms_per_batch=10)


def test_rejects_zero_max_atoms():
    with pytest.raises(ValueError, match="positive"):
        BucketBatchSampler([1, 2, 3], max_atoms_per_batch=0)


def test_rejects_zero_max_batch_size():
    with pytest.raises(ValueError, match="positive"):
        BucketBatchSampler([1, 2, 3], max_atoms_per_batch=10, max_batch_size=0)


def test_rejects_zero_bucket_size():
    with pytest.raises(ValueError, match="positive"):
        BucketBatchSampler([1, 2, 3], max_atoms_per_batch=10, bucket_size=0)


# ----------------------------- determinism -----------------------------------


def test_explicit_generator_makes_iteration_deterministic():
    lengths = list(range(5, 105))
    a = list(BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(123)
    ))
    b = list(BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(123)
    ))
    assert a == b


def test_distinct_seeds_produce_distinct_orderings():
    lengths = list(range(5, 105))
    a = list(BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(1)
    ))
    b = list(BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(2)
    ))
    assert a != b


def test_iteration_independent_of_global_numpy_seed():
    """The sampler must not be affected by global np.random state."""
    lengths = list(range(5, 105))
    np.random.seed(0)
    a = list(BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(99)
    ))
    np.random.seed(1234)
    b = list(BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(99)
    ))
    assert a == b


def test_two_iterations_differ_under_shuffle():
    """Calling __iter__ twice on the same sampler should yield different orderings."""
    lengths = list(range(5, 105))
    sampler = BucketBatchSampler(
        lengths, max_atoms_per_batch=200, generator=np.random.default_rng(7)
    )
    a = list(sampler)
    b = list(sampler)
    assert a != b


# ----------------------------- epoch-level mixing ----------------------------


def test_first_batch_is_not_always_smallest():
    """Bucket-order shuffle should mean the first batch isn't always smallest."""
    lengths = list(range(1, 201))
    first_batch_means = []
    for seed in range(20):
        sampler = BucketBatchSampler(
            lengths,
            max_atoms_per_batch=50,
            bucket_size=10,
            shuffle=True,
            generator=np.random.default_rng(seed),
        )
        first = next(iter(sampler))
        first_batch_means.append(np.mean([lengths[i] for i in first]))
    assert max(first_batch_means) - min(first_batch_means) > 50


# ----------------------------- bucket isolation (compile-friendliness) -------


def test_batches_do_not_cross_bucket_boundaries():
    """Every batch's items must come from a single bucket (similar lengths)."""
    lengths = list(range(1, 201))
    bucket_size = 20
    sampler = BucketBatchSampler(
        lengths,
        max_atoms_per_batch=80,
        bucket_size=bucket_size,
        shuffle=True,
        generator=np.random.default_rng(0),
    )
    sorted_order = np.argsort(lengths, kind="stable")
    bucket_of = {}
    for bucket_id, start in enumerate(range(0, len(sorted_order), bucket_size)):
        for i in sorted_order[start : start + bucket_size]:
            bucket_of[int(i)] = bucket_id

    for batch in sampler:
        if len(batch) <= 1:
            continue
        ids = {bucket_of[i] for i in batch}
        assert len(ids) == 1, (
            f"Batch crosses buckets: {batch} -> bucket ids {ids}"
        )


# ----------------------------- length self-consistency -----------------------


def test_len_matches_iteration_no_shuffle():
    sampler = _make_sampler(
        [5, 10, 15, 8, 3, 20, 12, 7], max_atoms_per_batch=25, shuffle=False
    )
    assert len(sampler) == len(list(sampler))


def test_len_exactly_matches_iteration_with_shuffle():
    """Bucket-local packing makes batch count shuffle-invariant."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(low=2, high=40, size=300).tolist()
    sampler = BucketBatchSampler(
        lengths,
        max_atoms_per_batch=120,
        bucket_size=64,
        shuffle=True,
        generator=np.random.default_rng(42),
    )
    assert len(sampler) == len(list(sampler))


# =============================== lengths_from_ptr ============================


def test_lengths_from_ptr_basic():
    assert lengths_from_ptr(np.array([0, 5, 12, 18])).tolist() == [5, 7, 6]


def test_lengths_from_ptr_single_structure():
    assert lengths_from_ptr(np.array([0, 7])).tolist() == [7]


def test_lengths_from_ptr_empty_returns_empty():
    assert lengths_from_ptr(np.array([0])).tolist() == []


def test_lengths_from_ptr_accepts_list():
    assert lengths_from_ptr([0, 3, 8]).tolist() == [3, 5]


def test_lengths_from_ptr_dtype_is_int64():
    out = lengths_from_ptr(np.array([0, 3, 8], dtype=np.int32))
    assert out.dtype == np.int64


# ============================ quantize_pad_length ============================


def test_quantize_pad_length_rounds_up():
    assert quantize_pad_length(67, 8) == 72
    assert quantize_pad_length(64, 8) == 64
    assert quantize_pad_length(1, 8) == 8


def test_quantize_pad_length_rejects_zero_multiple():
    with pytest.raises(ValueError, match="positive"):
        quantize_pad_length(10, 0)
