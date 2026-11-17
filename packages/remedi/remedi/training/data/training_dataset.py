from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from remedi.training.data.sample import Sample
from remedi_data.dataset import StructureFile
from remedi_data.format import STRUCTURES_FILENAME, read_table_polars

GetItemFn = Callable[["TrainingMoleculeDataset", int], Sample]


@dataclass(frozen=True)
class TrainingStructure:
    """One structure as training reads it."""

    atomic_numbers: np.ndarray
    positions: np.ndarray
    total_charge: float
    multiplicity: float


def atoms_getitem(ds: TrainingMoleculeDataset, i: int) -> Sample:
    structure = ds.structure(i)
    return Sample(
        atomic_positions=torch.from_numpy(structure.positions),
        atomic_numbers=torch.from_numpy(structure.atomic_numbers),
        total_charge=torch.tensor(structure.total_charge),
        multiplicity=torch.tensor(structure.multiplicity),
    )


def make_supervised_getitem(
    labels: np.ndarray, label_dtype: torch.dtype = torch.long
) -> GetItemFn:
    """Build a get-item that augments :func:`atoms_getitem` with a per-structure
    supervised label.

    ``labels`` is indexed by the same structure index the dataset uses (the
    TrainingMoleculeDataset shares the dataset's structure order), so
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

    ``structure_ids`` (sorted, unique) restricts the dataset to those
    structures, indexed ``0..len - 1`` in that order. Every sample is read
    from ``structures.atp`` when it is asked for (a memory-mapped record, a few
    microseconds); each process (each data-loader worker) opens its own
    :class:`StructureFile`. The charges and multiplicities of the structures
    (8 bytes each) are read from the table here, in the constructing process,
    and travel with the dataset to the workers.

    No table is read on the sample path: a forked loader worker that calls
    polars deadlocks, because the parent's polars thread pool (and the locks
    it holds) are not carried into the child.
    """

    def __init__(
        self,
        root: Path,
        get_item: GetItemFn,
        structure_ids: np.ndarray | None = None,
    ):
        self.root = Path(root)
        self._get_item = get_item
        self._structure_ids = (
            None if structure_ids is None else np.asarray(structure_ids, np.int64)
        )
        # Per process: the open file and the process that opened it.
        self._file: StructureFile | None = None
        self._file_process: int | None = None
        table = read_table_polars(self.root, columns=["total_charge", "multiplicity"])
        total_charge = table["total_charge"].to_numpy().astype(np.float32)
        multiplicity = table["multiplicity"].to_numpy().astype(np.float32)
        if self._structure_ids is not None:
            total_charge = total_charge[self._structure_ids]
            multiplicity = multiplicity[self._structure_ids]
        self._total_charge = total_charge
        self._multiplicity = multiplicity

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_file=None, _file_process=None)
        return state

    def _structure_file(self) -> StructureFile:
        if self._file is None or self._file_process != os.getpid():
            self._file = StructureFile.open(self.root / STRUCTURES_FILENAME)
            self._file_process = os.getpid()
        return self._file

    def open(self) -> TrainingMoleculeDataset:
        """Open the structure file now (in this process); returns the dataset."""
        self._structure_file()
        return self

    def atom_counts(self) -> np.ndarray:
        """Atoms per structure, in dataset order."""
        return self._structure_file().atom_counts(self._structure_ids)

    def structure(self, index: int) -> TrainingStructure:
        """Structure ``index`` of this dataset (``structure_ids[index]`` of the file)."""
        record = index if self._structure_ids is None else self._structure_ids[index]
        atomic_numbers, positions = self._structure_file().frame(int(record))
        return TrainingStructure(
            atomic_numbers=atomic_numbers,
            positions=positions,
            total_charge=float(self._total_charge[index]),
            multiplicity=float(self._multiplicity[index]),
        )

    def __len__(self):
        if self._structure_ids is not None:
            return len(self._structure_ids)
        return len(self._structure_file())

    def __getitem__(self, index: int) -> Sample:
        return self._get_item(self, int(index))
