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
    # Raw SMILES strings yielded by SMILES-text generators (TDC, Polaris,
    # SmilesList, TSV). ``FilterMoleculeStage`` consumes this list,
    # produces ``smiles: list[SmilesData]`` from the survivors, and clears
    # ``raw_smiles`` back to None. ``None`` entries inside the list are
    # treated as invalid rows by the stage. Generators that emit
    # pre-filtered SmilesData (e.g. MoleculeNet, which couples
    # filter+scaffold-split) leave this as None at the batch level.
    raw_smiles: list[str | None] | None = None

    def __len__(self):
        if self.molecules is not None:
            return len(self.molecules)
        if self.smiles is not None:
            return len(self.smiles)
        if self.raw_smiles is not None:
            return len(self.raw_smiles)
        return 0


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
