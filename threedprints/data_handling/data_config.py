from dataclasses import dataclass
from typing import Optional, Literal


DatasetTypes = Literal["Regression", "Evaluation", "Pretraining"]

@dataclass
class DatasetConfig:
    target_cols: list[str]
    N_molecules: int
    max_atoms: Optional[int]
    embedding_size: int
    dataset_type: DatasetTypes
    BFGS_tol: int
    BFGS_max_steps: int

