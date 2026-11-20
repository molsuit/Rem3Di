"""Tests for BucketBatchSampler, lengths_from_ptr, and quantize_pad_length."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from remedi.training.data.samplers import (
    BucketBatchSampler,
    lengths_from_ptr,
    quantize_pad_length,
)


def _flatten(batches: list[list[int]]) -> list[int]:
    return [i for b in batches for i in b]


def _make_sampler(lengths, **kwargs):
    kwargs.setdefault("seed", 0)
    return BucketBatchSampler(lengths, **kwargs)


# ----------------------------- coverage --------------------------------------


def test_every_index_appears_exactly_once():
    rng = np.random.default_rng(0)
    lengths = rng.integers(low=5, high=80, size=200).tolist()
    sampler = _make_sampler(lengths, max_atoms_per_batch=200, shuffle=True)
    flat = _flatten(list(sampler))
    assert sorted(flat) == list(range(len(lengths)))


def test_coverage_no_shuffle():
    sampler = _make_sampler([5, 10, 15, 20, 25], max_atoms_per_batch=30, shuffle=False)
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
    sampler = _make_sampler(lengths, max_atoms_per_batch=120, shuffle=True)
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


# ----------------------------- packing ---------------------------------------


def test_packing_has_no_forced_breaks():
    """Equal lengths fill every batch to the cap: only the last one is partial."""
    sampler = _make_sampler([10] * 1000, max_atoms_per_batch=50, shuffle=True)
    batches = list(sampler)
    assert len(batches) == 200
    assert all(len(batch) == 5 for batch in batches)


def test_packing_is_shuffle_invariant_in_count():
    lengths = np.random.default_rng(3).integers(2, 40, size=2000).tolist()
    shuffled = _make_sampler(lengths, max_atoms_per_batch=120, shuffle=True)
    ordered = _make_sampler(lengths, max_atoms_per_batch=120, shuffle=False)
    assert len(list(shuffled)) == len(list(ordered)) == len(shuffled)


# ----------------------------- input validation ------------------------------


def test_accepts_ndarray_input():
    sampler = _make_sampler(np.array([3, 4, 5]), max_atoms_per_batch=10, shuffle=False)
    flat = _flatten(list(sampler))
    assert sorted(flat) == [0, 1, 2]


def test_rejects_zero_length():
    with pytest.raises(ValueError, match="positive"):
        _make_sampler([1, 0, 3], max_atoms_per_batch=10)


def test_rejects_negative_length():
    with pytest.raises(ValueError, match="positive"):
        _make_sampler([1, -1, 3], max_atoms_per_batch=10)


def test_rejects_zero_max_atoms():
    with pytest.raises(ValueError, match="positive"):
        _make_sampler([1, 2, 3], max_atoms_per_batch=0)


def test_rejects_zero_max_batch_size():
    with pytest.raises(ValueError, match="positive"):
        _make_sampler([1, 2, 3], max_atoms_per_batch=10, max_batch_size=0)


# ----------------------------- determinism -----------------------------------


def test_distinct_seeds_produce_distinct_orderings():
    lengths = list(range(5, 105))
    a = list(BucketBatchSampler(lengths, max_atoms_per_batch=200, seed=1))
    b = list(BucketBatchSampler(lengths, max_atoms_per_batch=200, seed=2))
    assert a != b


def test_iteration_independent_of_global_numpy_seed():
    """The sampler must not be affected by global np.random state."""
    lengths = list(range(5, 105))
    np.random.seed(0)
    a = list(BucketBatchSampler(lengths, max_atoms_per_batch=200, seed=99))
    np.random.seed(1234)
    b = list(BucketBatchSampler(lengths, max_atoms_per_batch=200, seed=99))
    assert a == b


def test_two_iterations_differ_under_shuffle():
    """Calling __iter__ twice on the same sampler should yield different orderings."""
    lengths = list(range(5, 105))
    sampler = BucketBatchSampler(lengths, max_atoms_per_batch=200, seed=7)
    a = list(sampler)
    b = list(sampler)
    assert a != b


# ----------------------------- epoch-level mixing ----------------------------


def test_first_batch_is_not_always_smallest():
    """The batch-order shuffle means the first batch is not always the smallest."""
    lengths = list(range(1, 201))
    first_batch_means = []
    for seed in range(20):
        sampler = BucketBatchSampler(
            lengths,
            max_atoms_per_batch=50,
            shuffle=True,
            seed=seed,
        )
        first = next(iter(sampler))
        first_batch_means.append(np.mean([lengths[i] for i in first]))
    assert max(first_batch_means) - min(first_batch_means) > 50


# ----------------------------- similar lengths (compile-friendliness) --------


def test_every_batch_is_a_contiguous_range_of_the_sorted_lengths():
    """A batch never mixes a short structure with a long one."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 60, size=400)
    sampler = BucketBatchSampler(lengths, max_atoms_per_batch=80, shuffle=True, seed=0)
    sorted_lengths = np.sort(lengths)
    batch_ranges = sorted(
        (int(lengths[batch].min()), int(lengths[batch].max())) for batch in sampler
    )
    for (_, previous_maximum), (minimum, _) in itertools.pairwise(batch_ranges):
        assert previous_maximum <= minimum
    assert batch_ranges[0][0] == sorted_lengths[0]


# ----------------------------- length self-consistency -----------------------


def test_len_matches_iteration_no_shuffle():
    sampler = _make_sampler(
        [5, 10, 15, 8, 3, 20, 12, 7], max_atoms_per_batch=25, shuffle=False
    )
    assert len(sampler) == len(list(sampler))


def test_len_exactly_matches_iteration_with_shuffle():
    """Packing the sorted lengths makes the batch count shuffle-invariant."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(low=2, high=40, size=300).tolist()
    sampler = BucketBatchSampler(
        lengths,
        max_atoms_per_batch=120,
        shuffle=True,
        seed=42,
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


# ----------------------------- per-epoch composition -------------------------


def test_batch_composition_changes_between_epochs():
    """Equal-length structures are regrouped every epoch, not always batched
    with the same neighbours."""
    lengths = [10] * 100
    sampler = _make_sampler(lengths, max_atoms_per_batch=50)
    first = {frozenset(batch) for batch in sampler}
    second = {frozenset(batch) for batch in sampler}
    assert first != second
    assert len(sampler) == len(first) == len(second) == 20


def test_epochs_are_reproducible_from_the_seed():
    lengths = np.random.default_rng(0).integers(2, 30, size=300).tolist()

    def two_epochs(seed: int) -> list[list[list[int]]]:
        sampler = _make_sampler(lengths, max_atoms_per_batch=90, seed=seed)
        return [list(sampler), list(sampler)]

    assert two_epochs(3) == two_epochs(3)
    assert two_epochs(3) != two_epochs(4)


def test_dataloader_bucket_batches_follow_the_seed():
    from torch.utils.data import Dataset

    from remedi.configuration.dataloader_config import (
        BucketBatchSamplingConfig,
        DataLoaderConfig,
    )

    class _Indices(Dataset):
        def __len__(self) -> int:
            return 200

        def __getitem__(self, index: int) -> int:
            return index

    config = DataLoaderConfig(
        batch_sampling=BucketBatchSamplingConfig(max_atoms_per_batch=40),
        num_workers=0,
        pin_memory=False,
    )
    lengths = np.full(200, 8)

    def batches(seed: int) -> list[list[int]]:
        loader = config.build(_Indices(), lengths=lengths, collate_fn=list, seed=seed)
        return [batch for batch in loader]

    assert batches(1) == batches(1)
    assert batches(1) != batches(2)


def test_validation_loaders_can_drop_persistent_workers():
    from torch.utils.data import Dataset

    from remedi.configuration.dataloader_config import (
        BucketBatchSamplingConfig,
        DataLoaderConfig,
    )

    class _Indices(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, index: int) -> int:
            return index

    config = DataLoaderConfig(
        batch_sampling=BucketBatchSamplingConfig(max_atoms_per_batch=40),
        num_workers=1,
        pin_memory=False,
    )
    lengths = np.full(10, 8)
    training = config.build(_Indices(), lengths=lengths, collate_fn=list)
    validation = config.build(
        _Indices(), lengths=lengths, collate_fn=list, persistent_workers=False
    )
    assert training.persistent_workers
    assert not validation.persistent_workers
