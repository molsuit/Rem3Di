from __future__ import annotations

import math
import warnings
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """Length-bucketed batch sampler with deterministic, torch.compile-friendly shapes.

    Pipeline per epoch:
      1. Sort indices by length, split into contiguous buckets of `bucket_size`.
      2. Pack each bucket independently into batches under
         `max_atoms_per_batch` (sum-of-lengths cap) and optional `max_batch_size`.
         Packing is deterministic per bucket, so `__len__` is exact.
      3. Shuffle the order of all yielded batches.

    Determinism note: items in a bucket are tightly clustered by length, so the
    composition of each batch is fixed across epochs; only the order in which
    batches are visited varies. Across-bucket batch-order shuffle is what
    provides epoch-level mixing.
    """

    def __init__(
        self,
        lengths,
        max_atoms_per_batch: int,
        max_batch_size: int | None = None,
        bucket_size: int = 512,
        shuffle: bool = True,
        generator: np.random.Generator | None = None,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if (self.lengths <= 0).any():
            raise ValueError("All lengths must be positive.")

        self.max_atoms_per_batch = int(max_atoms_per_batch)
        if self.max_atoms_per_batch <= 0:
            raise ValueError("max_atoms_per_batch must be positive.")

        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when set.")
        self.max_batch_size = (
            int(max_batch_size) if max_batch_size is not None else None
        )

        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive.")
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self._rng = generator if generator is not None else np.random.default_rng()

        if self.lengths.size > 0:
            n_oversize = int((self.lengths > self.max_atoms_per_batch).sum())
            if n_oversize > 0:
                warnings.warn(
                    f"{n_oversize} sample(s) exceed max_atoms_per_batch="
                    f"{self.max_atoms_per_batch}; they will be yielded as singleton "
                    "batches.",
                    stacklevel=2,
                )

        self._sorted_buckets: list[np.ndarray] = self._build_sorted_buckets()
        self._batch_count: int = self._compute_batch_count()

    def _build_sorted_buckets(self) -> list[np.ndarray]:
        if self.lengths.size == 0:
            return []
        order = np.argsort(self.lengths, kind="stable")
        return [
            order[i : i + self.bucket_size]
            for i in range(0, order.size, self.bucket_size)
        ]

    def _pack_bucket(self, indices: np.ndarray) -> list[list[int]]:
        """Greedy-pack a single bucket of indices into batches."""
        batches: list[list[int]] = []
        current: list[int] = []
        atoms = 0

        for idx in indices:
            i = int(idx)
            length = int(self.lengths[i])

            if length > self.max_atoms_per_batch:
                if current:
                    batches.append(current)
                    current = []
                    atoms = 0
                batches.append([i])
                continue

            would_exceed_atoms = atoms + length > self.max_atoms_per_batch
            would_exceed_count = (
                self.max_batch_size is not None and len(current) >= self.max_batch_size
            )
            if current and (would_exceed_atoms or would_exceed_count):
                batches.append(current)
                current = []
                atoms = 0

            current.append(i)
            atoms += length

        if current:
            batches.append(current)
        return batches

    def _compute_batch_count(self) -> int:
        return sum(len(self._pack_bucket(b)) for b in self._sorted_buckets)

    def __iter__(self) -> Iterator[list[int]]:
        if not self._sorted_buckets:
            return

        all_batches: list[list[int]] = []
        for b in self._sorted_buckets:
            all_batches.extend(self._pack_bucket(b))

        if self.shuffle:
            self._rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        return self._batch_count


def lengths_from_ptr(ptr) -> np.ndarray:
    """Compute per-structure lengths from a ragged pointer array.

    Expects `ptr` of shape (n_structures + 1,) where lengths[i] = ptr[i+1]-ptr[i].
    """
    arr = np.asarray(ptr)
    return (arr[1:] - arr[:-1]).astype(np.int64, copy=False)


def quantize_pad_length(max_length: int, multiple: int) -> int:
    """Round `max_length` up to the next multiple of `multiple`.

    Use in collate functions to stabilize padded shapes for torch.compile:
    instead of padding every batch to its own max, pad to a small set of
    quantized lengths so the compiled graph cache is bounded.
    """
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    return int(math.ceil(max_length / multiple) * multiple)
