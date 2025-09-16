from pathlib import Path

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
    benchmark_loader,
    worker_init_fn,
)

torch.manual_seed(1)
np.random.seed(1)


dataset_path = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/pcqm"
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
    prefetch_factor=8,
    persistent_workers=True,
    pin_memory=True,
    num_workers=32,
    shuffle=True,
    collate_fn=pretraining_padded_collate_fn,
)






res = benchmark_loader(dataloader)
print(res)
