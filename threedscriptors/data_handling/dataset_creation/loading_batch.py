from dataclasses import dataclass

import numpy as np
import torch
from ase import Atoms

from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


@dataclass(slots=True)
class RegressionData:
    targets_system: np.ndarray | None = None
    mask_system: np.ndarray | None = None
    targets_atom: np.ndarray | None = None
    mask_atom: np.ndarray | None = None
    # Per-structure split codes (see Split enum). Rides inside RegressionData
    # so ConformerGenerationStage replicates it per conformer for free.
    split: np.ndarray | None = None


@dataclass
class SmilesData:
    nonisomeric_smiles: str
    isomeric_smiles: str


@dataclass
class InputBatch:
    smiles: list[SmilesData] | None
    molecules: list[Atoms] | None
    structure_ids: list[StructureID]
    total_charge: list[float] | None = None
    multiplicity: list[float] | None = None
    regression_data: RegressionData | None = None

    def __len__(self):
        if self.molecules is None:
            return 0
        return len(self.molecules)


@dataclass
class DataBatch:
    systems_index: torch.Tensor
    atomic_positions: torch.Tensor
    atomic_numbers: torch.Tensor
    structure_ids: list[StructureID]
    smiles_data: list[SmilesData] | None
    total_charge: torch.Tensor
    multiplicity: torch.Tensor
    regression_data: RegressionData | None = None
