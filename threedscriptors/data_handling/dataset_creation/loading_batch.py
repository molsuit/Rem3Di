from dataclasses import dataclass

import numpy as np
import torch
from ase import Atoms

from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


@dataclass(slots=True)
class RegressionData:
    regression_targets: np.ndarray
    regression_masks: np.ndarray


@dataclass
class SmilesData:
    nonisomeric_smiles: str
    isomeric_smiles: str


@dataclass
class InputBatch:
    smiles: list[SmilesData] | None
    molecules: list[Atoms] | None
    structure_ids: list[StructureID]
    regression_data: RegressionData | None = None

    def __len__(self):
        if self.molecules is None:
            return 0
        return len(self.molecules)


@dataclass
class DataBatch:
    embeddings: torch.Tensor
    systems_index: torch.Tensor
    atomic_positions: torch.Tensor
    atomic_numbers: torch.Tensor
    structure_ids: list[StructureID]
    smiles_data: list[SmilesData] | None
