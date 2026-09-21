"""Run a trained REM3DI encoder over a dataset and collect its descriptors.

The one survivor of the retired evaluation-utilities module: the scripts in
``scripts/latent_evaluation/`` need a plain ``(N, L * d_out)`` descriptor matrix
for a :class:`TrainingMoleculeDataset`, which is not what the benchmark
framework's disk-cached :func:`compute_and_cache` path produces. It stays here
rather than in ``evaluation/`` because descriptor-space analysis is what
consumes it.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from remedi.data_handling.sample import (
    Sample,
    yield_molecules_collate_fn,
)
from remedi.model.remedi_model import REM3DIModel


def evaluate_molecular_descriptor_on_dataset(
    model: REM3DIModel,
    dataset: Dataset,
    device="cuda",
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
):
    """Run the encoder over `dataset` and return descriptors as a flat
    `(N, L * d_out)` tensor — `L` seed tokens are concatenated per molecule.

    Raise ``num_workers`` to overlap zarr reads / sample featurization with the
    encoder forward pass; the existing TrainingMoleculeDataset rebuilds its
    zarr handles per-worker via ``__getstate__``/``__setstate__``.
    """
    batch_size = min(batch_size, len(dataset))

    dataloader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": yield_molecules_collate_fn,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader: Iterable[Sample] = DataLoader(dataset, **dataloader_kwargs)

    model.to(device)
    model.eval()

    aggregator = model.encoder.aggregator
    flat_dim = aggregator.seq_len * aggregator.d_out
    descriptors = torch.zeros(size=(len(dataset), flat_dim))

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            samples.to_(device)

            model_output = model(samples)
            descriptors[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                model_output.molecular_descriptor.flat
            )

    return descriptors
