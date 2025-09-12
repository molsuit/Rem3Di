import statistics as stats
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.data_handling.sample import pretraining_padded_collate_fn
from threedscriptors.training.data import (
    DatasetSplitting,
    SplitConfig,
    SplitStrategy,
    worker_init_fn,
)

torch.manual_seed(1)
np.random.seed(1)


dataset_path = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/pcqm"
)

get_item_fn = pos_emb_getitem
ds = TrainingMoleculeDataset(dataset_path, get_item_fn)


full_dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_path)

split_config = SplitConfig(strategy=SplitStrategy.SINGLE)
splitting = DatasetSplitting(full_dataset)

train_idx, val_idx, _ = next(splitting.get_split(split_config))

train_dataset = Subset(ds, train_idx)

dataloader = DataLoader(
    train_dataset,
    batch_size=512,
    worker_init_fn=worker_init_fn,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    num_workers=8,
    shuffle=True,
    collate_fn=pretraining_padded_collate_fn,
)



def benchmark_loader(dl, warmup=10, max_batches=100, device="cuda"):
    data_times, h2d_times, batch_sizes = [], [], []
    it = iter(dl)

    # warmup
    for _ in range(warmup):
        t0 = perf_counter()
        batch = next(it)
        data_wait = perf_counter() - t0

        t1 = perf_counter()
        batch = batch.to_(device=device, non_blocking=True)  # your .to_ method
        torch.cuda.synchronize()
        h2d = perf_counter() - t1

    # measure
    for _ in range(max_batches):
        t0 = perf_counter()
        batch = next(it)
        data_wait = perf_counter() - t0

        t1 = perf_counter()
        batch = batch.to_(device=device, non_blocking=True)
        torch.cuda.synchronize()
        h2d = perf_counter() - t1

        data_times.append(data_wait)
        h2d_times.append(h2d)
        try:
            batch_sizes.append(len(batch.padding_mask))  # or whatever equals batch size
        except Exception:
            batch_sizes.append(None)

    def summarise(xs):
        return dict(
            mean=stats.mean(xs),
            p50=stats.median(xs),
            p95=sorted(xs)[int(0.95 * len(xs)) - 1],
        )

    return {
        "batches": len(data_times),
        "data_wait": summarise(data_times),
        "h2d": summarise(h2d_times),
        "items_per_s": (
            None if batch_sizes[0] is None else sum(batch_sizes) / sum(data_times)
        ),
    }


res = benchmark_loader(dataloader)
print(res)
