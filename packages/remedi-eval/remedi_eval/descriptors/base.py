"""The descriptor-calculator interface.

A *descriptor calculator* turns a prepared :class:`MoleculeDataset` into an
``(N, D)`` matrix aligned with the dataset's structure order. Calculators are
fit-free (deterministic or pre-trained): one matrix per dataset is cached and
shared by every split, so a calculator that fitted on a train fold would leak
that fold into every other partition's cache entry.
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
    def calculate(self, dataset: MoleculeDataset) -> np.ndarray:
        """Return shape ``(dataset.N_structures, descriptor_dim)``."""
