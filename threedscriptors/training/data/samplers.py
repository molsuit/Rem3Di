from __future__ import annotations

from collections.abc import Iterator, Sequence

import math
import numpy as np
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    """
    Length-based bucketing for variable-size data (e.g. molecules).

    Constraints:
      - max_atoms_per_batch: upper bound on sum(num_atoms) in a batch.
      - max_batch_size: optional hard cap on number of samples per batch.

    Strategy:
      1) Sort samples by length.
      2) Shuffle within buckets of similar lengths.
      3) Greedily pack indices into batches under the constraints.
    """

    def __init__(
        self,
        lengths,
        max_atoms_per_batch: int,
        max_batch_size: int | None = None,
        bucket_size: int = 512,
        shuffle: bool = True,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        self.max_atoms_per_batch = int(max_atoms_per_batch)
        self.max_batch_size = int(max_batch_size) if max_batch_size is not None else None
        self.bucket_size = int(bucket_size)
        self.shuffle = shuffle

        assert (self.lengths > 0).all(), "All lengths must be positive."

    def __iter__(self):
        # 1) sort by length
        order = np.argsort(self.lengths)

        # 2) optional: shuffle within buckets to keep similar sizes together
        if self.shuffle:
            buckets = [
                order[i : i + self.bucket_size]
                for i in range(0, len(order), self.bucket_size)
            ]
            for b in buckets:
                np.random.shuffle(b)
            order = np.concatenate(buckets)

        # 3) greedy packing
        batch = []
        atoms_in_batch = 0

        for idx in order:
            L = int(self.lengths[idx])
            if L > self.max_atoms_per_batch:
                # Single huge sample: yield alone to avoid infinite loop.
                if batch:
                    yield batch
                    batch = []
                    atoms_in_batch = 0
                yield [int(idx)]
                continue

            new_atoms = atoms_in_batch + L
            too_many_atoms = new_atoms > self.max_atoms_per_batch
            too_many_samples = (
                self.max_batch_size is not None
                and len(batch) >= self.max_batch_size
            )

            if batch and (too_many_atoms or too_many_samples):
                yield batch
                batch = []
                atoms_in_batch = 0

            batch.append(int(idx))
            atoms_in_batch += L

        if batch:
            yield batch

    def __len__(self):
        # Deterministic estimate: compute without shuffle
        order = np.argsort(self.lengths)
        n_batches = 0
        atoms_in_batch = 0
        batch_size = 0
        for idx in order:
            L = int(self.lengths[idx])
            if L > self.max_atoms_per_batch:
                if batch_size > 0:
                    n_batches += 1
                    batch_size = 0
                    atoms_in_batch = 0
                n_batches += 1
                continue

            new_atoms = atoms_in_batch + L
            too_many_atoms = new_atoms > self.max_atoms_per_batch
            too_many_samples = (
                self.max_batch_size is not None
                and batch_size >= self.max_batch_size
            )
            if batch_size > 0 and (too_many_atoms or too_many_samples):
                n_batches += 1
                atoms_in_batch = 0
                batch_size = 0
            atoms_in_batch += L
            batch_size += 1
        if batch_size > 0:
            n_batches += 1
        return n_batches


def lengths_from_ptr(ptr: np.ndarray) -> np.ndarray:
    """Compute per-structure lengths from ragged pointer array.

    Expects `ptr` of shape (n_structures + 1,) where lengths[i] = ptr[i+1]-ptr[i].
    Returns an array of shape (n_structures,).
    """
    if not isinstance(ptr, np.ndarray):
        ptr = np.asarray(ptr)
    return (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)

