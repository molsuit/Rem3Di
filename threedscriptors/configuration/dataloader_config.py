from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset


class RandomShuffleSamplingConfig(BaseModel):
    kind: Literal["random_shuffle"] = "random_shuffle"
    batch_size: int = Field(gt=0)
    drop_last: bool = False


class BucketBatchSamplingConfig(BaseModel):
    kind: Literal["bucket_batch"] = "bucket_batch"
    max_atoms_per_batch: int = Field(gt=0)
    max_batch_size: int | None = Field(default=None, gt=0)
    bucket_size: int = Field(default=512, gt=0)


BatchSamplingConfig = Annotated[
    RandomShuffleSamplingConfig | BucketBatchSamplingConfig,
    Field(discriminator="kind"),
]


class DataLoaderConfig(BaseModel):
    batch_sampling: BatchSamplingConfig
    num_workers: int = Field(default=12, ge=0)
    prefetch_factor: int | None = Field(default=4, ge=1)
    persistent_workers: bool = True
    pin_memory: bool = True

    def build(
        self,
        dataset: Dataset,
        *,
        lengths: np.ndarray | None = None,
        collate_fn=None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> DataLoader:
        """Construct a DataLoader from this config.

        `lengths` is required when batch_sampling is bucket_batch; ignored
        otherwise. `shuffle` is honored by both sampling kinds (random shuffle
        vs bucket-batch-order shuffle).
        """
        from threedscriptors.training.data.samplers import BucketBatchSampler
        from threedscriptors.training.data.util import worker_init_fn

        prefetch_factor = (
            self.prefetch_factor if self.num_workers > 0 else None
        )
        persistent_workers = self.persistent_workers and self.num_workers > 0

        sampling = self.batch_sampling
        if isinstance(sampling, RandomShuffleSamplingConfig):
            return DataLoader(
                dataset,
                batch_size=sampling.batch_size,
                shuffle=shuffle,
                drop_last=sampling.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )

        if lengths is None:
            raise ValueError(
                "BucketBatchSampling requires per-sample lengths to be provided."
            )
        rng = np.random.default_rng(seed) if seed is not None else None
        batch_sampler = BucketBatchSampler(
            lengths,
            max_atoms_per_batch=sampling.max_atoms_per_batch,
            max_batch_size=sampling.max_batch_size,
            bucket_size=sampling.bucket_size,
            shuffle=shuffle,
            generator=rng,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
