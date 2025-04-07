from enum import Enum

from pydantic import BaseModel, ConfigDict

from threedscriptors.configuration.config_utils import NumpyArrayType


class DatasetTypes(Enum):
    REGRESSION = 1
    EVALUATION = 2
    PRETRAINING = 3


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    N_molecules: int
    max_atoms: int | None
    dataset_type: DatasetTypes
    BFGS_tol: int
    BFGS_max_steps: int
    N_conformers: int
    target_cols: list[str] | None
    mean: NumpyArrayType | None = None
    std: NumpyArrayType | None = None
