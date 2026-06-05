from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.sample import Sample

GetItemFn = Callable[["TrainingMoleculeDataset", int], Sample]


def atoms_getitem(ds: "TrainingMoleculeDataset", i: int) -> Sample:
    a0, a1 = ds._ptr[i : i + 2].tolist()
    pos = torch.from_numpy(np.asarray(ds._pos[a0:a1]))
    num = torch.from_numpy(np.asarray(ds._atom_num[a0:a1]))
    charge = torch.tensor(float(ds._charge[i]))
    mult = torch.tensor(float(ds._multiplicity[i]))
    return Sample(
        atomic_positions=pos,
        atomic_numbers=num,
        total_charge=charge,
        multiplicity=mult,
    )


def make_supervised_getitem(
    labels: np.ndarray, label_dtype: torch.dtype = torch.long
) -> GetItemFn:
    """Build a get-item that augments :func:`atoms_getitem` with a per-structure
    supervised label.

    ``labels`` is indexed by the same structure index the dataset uses (the
    TrainingMoleculeDataset shares the source zarr's ordering), so
    ``labels[i]`` is the target for structure ``i``. The label is attached as a
    scalar ``Sample.regression_targets`` tensor (``long`` for classification
    class indices, ``float`` for regression), which
    :func:`yield_molecules_supervised_collate_fn` stacks into a ``(B,)`` batch.
    """

    def _getitem(ds: "TrainingMoleculeDataset", i: int) -> Sample:
        sample = atoms_getitem(ds, i)
        sample.regression_targets = torch.tensor(labels[i], dtype=label_dtype)
        return sample

    return _getitem


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
        self._multiplicity = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(
            _group=None,
            _ptr=None,
            _pos=None,
            _atom_num=None,
            _charge=None,
            _multiplicity=None,
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
            self._multiplicity = np.array(g["multiplicity"])
            self._group = None
        else:
            # zarr v3 sharded reads pull a byte-range out of a shard file;
            # the OS page cache covers repeated access to hot shards (zarr v3
            # has no LRUStoreCache equivalent).
            g = zarr.open_group(self.root, mode="r")
            self._group = g

            self._ptr = g["molecule_ptr"]
            self._pos = g["positions"]
            self._atom_num = g["atomic_numbers"]
            self._charge = g["total_charge"]
            self._multiplicity = g["multiplicity"]

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
        return cls(Path(dataset.path), get_item=get_item)
