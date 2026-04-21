from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from zarr import DirectoryStore, LRUStoreCache

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.sample import Sample

GetItemFn = Callable[["TrainingMoleculeDataset", int], Sample]


def pos_emb_getitem(ds: "TrainingMoleculeDataset", i: int) -> Sample:
    # Read both bounds in one go (one chunk decompress)
    a0, a1 = ds._ptr[i : i + 2].tolist()
    emb_np = ds._emb[a0:a1]  # zarr -> numpy view/copy as needed
    pos_np = ds._pos[a0:a1]
    # Zero-copy into torch where possible
    emb = torch.from_numpy(emb_np)
    pos = torch.from_numpy(pos_np)
    return Sample(embeddings=emb, atomic_positions=pos)


def atoms_getitem(ds: "TrainingMoleculeDataset", i: int):
    a0, a1 = ds._ptr[i : i + 2].tolist()
    num_np = ds._atom_num[a0:a1]  # zarr -> numpy view/copy as needed
    pos_np = ds._pos[a0:a1]

    pos = torch.from_numpy(pos_np)
    num = torch.from_numpy(num_np)
    return Sample(atomic_positions=pos, atomic_numbers=num)


class TrainingMoleculeDataset(Dataset):
    def __init__(self, root: Path, get_item: GetItemFn, in_memory: bool = False):
        self.root = str(root)
        self._get_item = get_item
        self._in_memory = in_memory

        # process-local handles / arrays
        self._group = None
        self._ptr = None
        self._emb = None
        self._pos = None
        self._atom_num = None

    def __getstate__(self):
        d = dict(self.__dict__)
        # drop process-local references so each worker re-opens cleanly
        d.update(
            _group=None,
            _ptr=None,
            _emb=None,
            _pos=None,
            _atom_num=None,
        )
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # optionally eager-open for workers
        self._ensure_open()

    def _ensure_open(self):
        # already initialized
        if self._ptr is not None:
            return

        if self._in_memory:
            # load everything into RAM once (per process)
            g = zarr.open_group(self.root, mode="r")
            # materialize as numpy arrays
            self._ptr = np.array(g["molecule_ptr"])  # (N+1,)

            if "atomic_embeddings" in list(g.arrays()):
                self._emb = g["atomic_embeddings"]

            self._pos = np.array(g["positions"])  # (total_atoms, 3)
            self._atom_num = np.array(g["atomic_numbers"])  # (total_atoms,)
            self._group = None
        else:
            # on-disk zarr arrays with cached store
            store = DirectoryStore(self.root)
            store = LRUStoreCache(store, max_size=2**29)  # ~256 MiB per worker
            g = zarr.open_group(store=store, mode="r")
            self._group = g

            if "atomic_embeddings" in list(g.arrays()):
                self._emb = g["atomic_embeddings"]

            self._ptr = g["molecule_ptr"]
            self._pos = g["positions"]
            self._atom_num = g["atomic_numbers"]

    def __len__(self):
        self._ensure_open()
        return int(self._ptr.shape[0] - 1)

    def __getitem__(self, idx: int) -> Sample:
        self._ensure_open()
        return self._get_item(self, int(idx))

    @classmethod
    def from_molecule_dataset(
        cls,
        dataset: MoleculeDataset,
        *,
        get_item: GetItemFn = pos_emb_getitem,
    ) -> "TrainingMoleculeDataset":
        store = dataset.atomic_embeddings.store
        while hasattr(store, "store"):
            store = store.store
        if not isinstance(store, DirectoryStore):
            raise TypeError(
                f"TrainingMoleculeDataset requires a DirectoryStore, got {type(store)}"
            )
        return cls(Path(store.path), get_item=get_item)
