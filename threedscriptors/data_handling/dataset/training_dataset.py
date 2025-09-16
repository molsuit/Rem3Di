from collections.abc import Callable
from pathlib import Path

import torch
import zarr
from torch.utils.data import Dataset
from zarr import DirectoryStore, LRUStoreCache

from threedscriptors.data_handling.sample import Sample

GetItemFn = Callable[["TrainingMoleculeDataset", int], Sample]

def pos_emb_getitem(ds: "TrainingMoleculeDataset", i: int) -> Sample:
    # Read both bounds in one go (one chunk decompress)
    a0, a1 = ds._ptr[i:i+2].tolist()
    emb_np = ds._emb[a0:a1]       # zarr -> numpy view/copy as needed
    pos_np = ds._pos[a0:a1]
    # Zero-copy into torch where possible
    emb = torch.from_numpy(emb_np)
    pos = torch.from_numpy(pos_np)
    return Sample(embeddings=emb, atomic_positions=pos)

class TrainingMoleculeDataset(Dataset):
    def __init__(self, root: Path, get_item: GetItemFn):
        self.root = str(root)
        self._group = None        # opened per worker
        self._ptr = None
        self._emb = None
        self._pos = None
        self._get_item = get_item

    def __getstate__(self):
        d = dict(self.__dict__)
        # drop process-local handles before worker copy/pickle
        d.update(_group=None, _ptr=None, _emb=None, _pos=None)
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # optional: eager-open in worker so first batch is snappy
        self._ensure_open()

    def _ensure_open(self):
        if self._group is None:
            store = DirectoryStore(self.root)
            store = LRUStoreCache(store, max_size=2**29)  # ~256 MiB per worker
            g = zarr.open_group(store=store, mode="r")
            self._group = g
            self._ptr = g["molecule_ptr"]
            self._emb = g["atomic_embeddings"]
            self._pos = g["positions"]  # <- correct source for positions

    def __len__(self):
        self._ensure_open()
        return int(self._ptr.shape[0] - 1)

    def __getitem__(self, idx: int) -> Sample:
        self._ensure_open()
        return self._get_item(self, int(idx))
