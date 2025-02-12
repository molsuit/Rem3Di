from dataclasses import dataclass
from typing import Literal

from torch import Tensor

type DatasetTypes = Literal["Regression", "Evaluation", "Pretraining"]


@dataclass
class DatasetConfig:
    target_cols: list[str]
    N_molecules: int
    max_atoms: int | None
    embedding_size: int
    dataset_type: DatasetTypes
    BFGS_tol: int
    BFGS_max_steps: int
    chirality: bool
    N_conformers: int
    mean: Tensor | None = None
    std: Tensor | None = None
