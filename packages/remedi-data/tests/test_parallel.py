"""The shared process map keeps input order with and without a pool, and the
worker count follows the Slurm allocation."""

from __future__ import annotations

import os

import pytest

from remedi_data.parallel import default_workers, process_map, process_map_chunks


@pytest.mark.parametrize("workers", [1, 3])
def test_process_map_keeps_the_input_order(workers: int) -> None:
    items = list(range(-10, 10))
    assert process_map(abs, items, workers=workers, chunk_size=3) == [
        abs(item) for item in items
    ]


def test_process_map_chunks_hands_each_slice_to_the_function() -> None:
    items = list(range(10))
    assert process_map_chunks(sum, items, workers=2, chunk_size=4) == [6, 22, 17]


def test_no_items_give_no_results() -> None:
    assert process_map(abs, [], workers=4, chunk_size=10) == []


def test_a_chunk_size_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        process_map(abs, [1], workers=1, chunk_size=0)


def test_default_workers_follows_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    assert default_workers() == 7
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert default_workers() == len(os.sched_getaffinity(0))
