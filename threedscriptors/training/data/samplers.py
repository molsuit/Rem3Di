from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import BatchSampler


class BucketByLengthBatchSampler(BatchSampler):
    """
    Groups indices into batches of similar lengths to reduce padding.

    Typical usage: pass structure lengths (atoms per structure) and the
    subset indices to be batched. Batches are formed by sorting by length
    once and then chunking into fixed-size batches. Optionally shuffles
    the batch order each epoch while keeping elements within a batch similar.
    """

    def __init__(
        self,
        indices: Sequence[int],
        lengths: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        shuffle_batches: bool = True,
        seed: int | None = 1,
    ) -> None:
        if not isinstance(lengths, np.ndarray):
            lengths = np.asarray(lengths)

        self.indices = np.asarray(indices, dtype=np.int64)
        self.lengths = lengths.astype(np.int64, copy=False)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle_batches = bool(shuffle_batches)
        self._rng = np.random.default_rng(seed) if seed is not None else None

        # Precompute a stable sort by length (descending)
        order = np.argsort(self.lengths[self.indices])[::-1]
        self._sorted_indices = self.indices[order]

        # Materialise contiguous slices as batches
        n = len(self._sorted_indices)
        bs = self.batch_size
        self._batches: list[np.ndarray] = [
            self._sorted_indices[i : i + bs] for i in range(0, n, bs)
        ]
        if self.drop_last and (len(self._batches) > 0) and (
            len(self._batches[-1]) < bs
        ):
            self._batches.pop()

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle at the batch granularity each epoch
        if self.shuffle_batches and self._rng is not None and len(self._batches) > 1:
            perm = self._rng.permutation(len(self._batches))
            batches = [self._batches[i] for i in perm]
        else:
            batches = self._batches

        for b in batches:
            # Yield as Python list for DataLoader consumption
            yield b.tolist()

    def __len__(self) -> int:
        return len(self._batches)


def lengths_from_ptr(ptr: np.ndarray) -> np.ndarray:
    """Compute per-structure lengths from ragged pointer array.

    Expects `ptr` of shape (n_structures + 1,) where lengths[i] = ptr[i+1]-ptr[i].
    Returns an array of shape (n_structures,).
    """
    if not isinstance(ptr, np.ndarray):
        ptr = np.asarray(ptr)
    return (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)

