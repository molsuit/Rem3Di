from dataclasses import dataclass
from typing import Literal

import numpy as np

type DatasetTypes = Literal["Regression", "Evaluation", "Pretraining"]


@dataclass
class DatasetConfig:
    N_molecules: int
    max_atoms: int | None
    dataset_type: DatasetTypes
    BFGS_tol: int
    BFGS_max_steps: int
    N_conformers: int
    target_cols: list[str] | None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
