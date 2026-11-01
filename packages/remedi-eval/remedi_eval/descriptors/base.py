"""The descriptor-calculator interface.

A *descriptor calculator* turns a prepared :class:`MoleculeDataset` into an
``(N, D)`` matrix aligned with the dataset's structure order. The optional
``train_indices`` keyword is for future fitted descriptors (e.g. a learned PCA)
that must only fit on the train split; deterministic and pre-trained
calculators ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from remedi_data.dataset.molecule_dataset import MoleculeDataset


class DescriptorCalculator(ABC):
    """One method, one matrix.

    Concrete subclasses are responsible for *how* they read the dataset
    (SMILES vs DataLoader over the zarr) but must produce a result aligned
    with the dataset's structure order so the runner can slice it by the
    materialized ``split`` column.
    """

    @abstractmethod
    def calculate(
        self,
        dataset: MoleculeDataset,
        *,
        train_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return shape ``(dataset.N_structures, descriptor_dim)``."""
