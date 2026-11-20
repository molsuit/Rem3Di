from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """Length-bucketed batch sampler with deterministic, torch.compile-friendly shapes.

    Pipeline per epoch:
      1. With ``shuffle``, permute the indices with the epoch's generator, then
         stable-sort them by length. Structures of equal length land in a
         different order (so in different batches) every epoch, while the
         sorted lengths never change.
      2. Pack the sorted order greedily into batches under
         ``max_atoms_per_batch`` (sum-of-lengths cap) and optional
         ``max_batch_size``: each batch holds structures of nearly the same
         length (few padded atom slots, few distinct shapes for torch.compile).
         The packing depends only on the sorted lengths, so it is computed once
         and ``__len__`` is exact.
      3. With ``shuffle``, shuffle the order of the batches.

    Epoch ``e`` draws from ``numpy.random.default_rng([seed, e])``: a run is
    reproducible from ``seed`` and every epoch differs.
    """

    def __init__(
        self,
        lengths,
        max_atoms_per_batch: int,
        max_batch_size: int | None = None,
        shuffle: bool = True,
        *,
        seed: int,
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

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if self.lengths.size > 0:
            n_oversize = int((self.lengths > self.max_atoms_per_batch).sum())
            if n_oversize > 0:
                warnings.warn(
                    f"{n_oversize} sample(s) exceed max_atoms_per_batch="
                    f"{self.max_atoms_per_batch}; they will be yielded as singleton "
                    "batches.",
                    stacklevel=2,
                )

        self._batch_bounds = self._pack(np.sort(self.lengths, kind="stable"))

    def _pack(self, sorted_lengths: np.ndarray) -> list[tuple[int, int]]:
        """Greedy-pack the sorted lengths; ``(start, stop)`` positions in the sorted
        order. A structure above ``max_atoms_per_batch`` is a batch of its own."""
        bounds: list[tuple[int, int]] = []
        start = 0
        atoms = 0
        for position in range(sorted_lengths.size):
            length = int(sorted_lengths[position])
            if length > self.max_atoms_per_batch:
                if position > start:
                    bounds.append((start, position))
                bounds.append((position, position + 1))
                start = position + 1
                atoms = 0
                continue
            would_exceed_atoms = atoms + length > self.max_atoms_per_batch
            would_exceed_count = (
                self.max_batch_size is not None
                and position - start >= self.max_batch_size
            )
            if position > start and (would_exceed_atoms or would_exceed_count):
                bounds.append((start, position))
                start = position
                atoms = 0
            atoms += length
        if sorted_lengths.size > start:
            bounds.append((start, sorted_lengths.size))
        return bounds

    def set_epoch(self, epoch: int) -> None:
        """Make the next iteration draw epoch ``epoch``'s order (used to resume
        a run and after iterations that were not a training epoch)."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        if self.shuffle:
            permutation = generator.permutation(self.lengths.size)
            order = permutation[np.argsort(self.lengths[permutation], kind="stable")]
        else:
            order = np.argsort(self.lengths, kind="stable")
        batches = [order[start:stop].tolist() for start, stop in self._batch_bounds]
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batch_bounds)


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
    # Integer ceiling division: exact, and keeps a symbolic size symbolic under
    # torch.export (math.ceil would specialise it to the example's value).
    return -(-max_length // multiple) * multiple
