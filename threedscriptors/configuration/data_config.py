from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict


class DatasetTypes(Enum):
    REGRESSION = 1
    EVALUATION = 2
    PRETRAINING = 3


class TaskConfig(BaseModel):
    task_name: str
    mean: float | None = None
    std: float | None = None
    has_auxillary_data: bool = False
    auxillary_data_dimension: int | None = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    N_molecules: int
    dataset_type: DatasetTypes
    BFGS_tol: float
    BFGS_max_steps: int
    N_conformers: int = 1
    embedding_model: str | None = None
    max_atoms: int | None = None
    is_normalized: bool = False
    tasks: Sequence[TaskConfig] | None = None
    has_relaxed_positions: bool = False
    has_atomic_embeddings: bool = False
    reload_from_directory: str | None = None
