from collections.abc import Sequence
from enum import Enum

from mace.calculators import MACECalculator
from pydantic import BaseModel, ConfigDict, model_validator


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


class MaceCalculatorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mace_calc: MACECalculator
    model_name: str
    model_path: str | None = None
    enable_cueq: bool | None = None
    device: str | None = None

    @model_validator(mode="before")
    @classmethod
    def create_default_mace_calc(cls, data):
        if data["mace_calc"] is None:
            model_path = data.get("model_path")
            enable_cueq = data.get("enable_cueq", False)
            device = data.get("device", "cpu")

            if model_path is None:
                raise ValueError("Can not create Mace Model without specified values")
            data["mace_calc"] = MACECalculator(
                model_path, enable_cueq=enable_cueq, device=device
            )

        return data


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    N_molecules: int
    dataset_type: DatasetTypes
    BFGS_tol: float
    BFGS_max_steps: int
    N_conformers: int = 1
    embedding_model_config: MaceCalculatorConfig | None = None
    max_atoms: int | None = None
    is_normalized: bool = False
    tasks: Sequence[TaskConfig] | None = None
    has_relaxed_positions: bool = False
    has_atomic_embeddings: bool = False
    reload_from_directory: str | None = None
