"""Process pools: how many workers, and an ordered map over chunks of items.

:func:`process_map_chunks` is the pool for CPU-bound per-item work (SMILES
parsing, scaffolds, fingerprints, descriptors): the items are cut into chunks
of ``chunk_size``, each chunk is one task, and the results come back in input
order. One worker or one chunk runs in this process with no pool, which keeps
small inputs (and tests that monkeypatch the worker function) out of the
pickling boundary. The pool uses the ``forkserver`` start method, so the
function must be importable at module level (a ``functools.partial`` of one is
fine) and no state of the parent process leaks into the workers.

Work that needs per-task failure isolation (the structure-preparation shards,
the conformer pool that bisects worker crashes) keeps its own executor and
takes only :func:`default_workers` from here.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def default_workers() -> int:
    """``SLURM_CPUS_PER_TASK`` inside a Slurm job, else the CPUs this process may use."""
    from_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if from_slurm:
        return max(1, int(from_slurm))
    return max(1, len(os.sched_getaffinity(0)))


def process_map_chunks[Item, Result](
    function: Callable[[Sequence[Item]], Result],
    items: Sequence[Item],
    *,
    workers: int,
    chunk_size: int,
) -> list[Result]:
    """``function`` of each consecutive ``chunk_size`` slice of ``items``, in order.

    Raises:
        ValueError: if ``chunk_size`` is not positive.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    chunks = [
        items[start : start + chunk_size] for start in range(0, len(items), chunk_size)
    ]
    if workers <= 1 or len(chunks) <= 1:
        return [function(chunk) for chunk in chunks]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(chunks)),
        mp_context=multiprocessing.get_context("forkserver"),
    ) as executor:
        return list(executor.map(function, chunks))


def _apply_to_each[Item, Result](
    function: Callable[[Item], Result], chunk: Sequence[Item]
) -> list[Result]:
    return [function(item) for item in chunk]


def process_map[Item, Result](
    function: Callable[[Item], Result],
    items: Sequence[Item],
    *,
    workers: int,
    chunk_size: int,
) -> list[Result]:
    """``function`` of every item, in order, ``chunk_size`` items per pool task."""
    chunks = process_map_chunks(
        partial(_apply_to_each, function), items, workers=workers, chunk_size=chunk_size
    )
    return [result for chunk in chunks for result in chunk]
