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


def atoms_getitem(ds: "TrainingMoleculeDataset", i: int) -> Sample:
    a0, a1 = ds._ptr[i : i + 2].tolist()
    pos = torch.from_numpy(np.asarray(ds._pos[a0:a1]))
    num = torch.from_numpy(np.asarray(ds._atom_num[a0:a1]))
    charge = torch.tensor(float(ds._charge[i]))
    spin = torch.tensor(float(ds._spin[i]))
    return Sample(
        atomic_positions=pos,
        atomic_numbers=num,
        total_charge=charge,
        total_spin=spin,
    )


class TrainingMoleculeDataset(Dataset):
    def __init__(self, root: Path, get_item: GetItemFn, in_memory: bool = False):
        self.root = str(root)
        self._get_item = get_item
        self._in_memory = in_memory

        # process-local handles / arrays
        self._group = None
        self._ptr = None
        self._pos = None
        self._atom_num = None
        self._charge = None
        self._spin = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(
            _group=None,
            _ptr=None,
            _pos=None,
            _atom_num=None,
            _charge=None,
            _spin=None,
        )
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ensure_open()

    def _ensure_open(self):
        if self._ptr is not None:
            return

        if self._in_memory:
            g = zarr.open_group(self.root, mode="r")
            self._ptr = np.array(g["molecule_ptr"])
            self._pos = np.array(g["positions"])
            self._atom_num = np.array(g["atomic_numbers"])
            self._charge = np.array(g["total_charge"])
            self._spin = np.array(g["total_spin"])
            self._group = None
        else:
            store = DirectoryStore(self.root)
            store = LRUStoreCache(store, max_size=2**29)
            g = zarr.open_group(store=store, mode="r")
            self._group = g

            self._ptr = g["molecule_ptr"]
            self._pos = g["positions"]
            self._atom_num = g["atomic_numbers"]
            self._charge = g["total_charge"]
            self._spin = g["total_spin"]

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
        get_item: GetItemFn = atoms_getitem,
    ) -> "TrainingMoleculeDataset":
        store = dataset.positions.store
        while hasattr(store, "store"):
            store = store.store
        if not isinstance(store, DirectoryStore):
            raise TypeError(
                f"TrainingMoleculeDataset requires a DirectoryStore, got {type(store)}"
            )
        return cls(Path(store.path), get_item=get_item)
