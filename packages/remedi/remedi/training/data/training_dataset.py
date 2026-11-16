from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from remedi.training.data.sample import Sample
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.packed_structures import PackedStructures

GetItemFn = Callable[["TrainingMoleculeDataset", int], Sample]


def atoms_getitem(ds: TrainingMoleculeDataset, i: int) -> Sample:
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

    def _getitem(ds: TrainingMoleculeDataset, i: int) -> Sample:
        sample = atoms_getitem(ds, i)
        sample.regression_targets = torch.tensor(labels[i], dtype=label_dtype)
        return sample

    return _getitem


class TrainingMoleculeDataset(Dataset):
    """The structures of one dataset, or of a subset of it, as training samples.

    ``structure_ids`` (sorted, unique) restricts the dataset to those structures,
    gathered into compact in-memory arrays and indexed ``0..len - 1`` in that
    order. Call :meth:`open` before handing the dataset to a forking data
    loader, so the workers inherit the arrays instead of each reading them.
    ``arrays`` hands over structures already gathered for ``structure_ids``
    (they are re-gathered only after pickling).
    """

    def __init__(
        self,
        root: Path,
        get_item: GetItemFn,
        in_memory: bool = False,
        structure_ids: np.ndarray | None = None,
        arrays: PackedStructures | None = None,
    ):
        self.root = str(root)
        self._get_item = get_item
        self._in_memory = in_memory
        self._structure_ids = (
            None if structure_ids is None else np.asarray(structure_ids, np.int64)
        )

        # process-local handles / arrays
        self._ptr = None
        self._pos = None
        self._atom_num = None
        self._charge = None
        self._multiplicity = None
        if arrays is not None:
            if self._structure_ids is None:
                raise ValueError("arrays need the structure_ids they were gathered for")
            self._set_arrays(arrays)

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(
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

        dataset = MoleculeDataset.open_existing_dataset_from_dir(Path(self.root))
        if self._in_memory or self._structure_ids is not None:
            self._set_arrays(dataset.read_structures(self._structure_ids))
        else:
            # Lazy: the zarr arrays themselves, read per sample.
            self._ptr = dataset.ptr
            self._pos = dataset.positions
            self._atom_num = dataset.atomic_numbers
            self._charge = dataset.total_charge
            self._multiplicity = dataset.multiplicity

    def _set_arrays(self, arrays: PackedStructures) -> None:
        if arrays.total_charge is None or arrays.multiplicity is None:
            raise ValueError("training structures need charges and multiplicities")
        self._ptr = arrays.ptr
        self._pos = arrays.positions
        self._atom_num = arrays.atomic_numbers
        self._charge = arrays.total_charge
        self._multiplicity = arrays.multiplicity

    def open(self) -> TrainingMoleculeDataset:
        """Read the arrays now (in this process); returns the dataset."""
        self._ensure_open()
        return self

    def atom_counts(self) -> np.ndarray:
        """Atoms per structure, in dataset order."""
        self._ensure_open()
        return np.diff(np.asarray(self._ptr)).astype(np.int64)

    def __len__(self):
        self._ensure_open()
        return int(self._ptr.shape[0] - 1)

    def __getitem__(self, idx: int) -> Sample:
        self._ensure_open()
        return self._get_item(self, int(idx))
