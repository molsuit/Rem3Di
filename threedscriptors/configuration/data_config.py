from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import torch
from mace.calculators import MACECalculator
from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


class DatasetTypes(Enum):
    REGRESSION = 1
    EVALUATION = 2
    PRETRAINING = 3
    SIMILARITY_SCREENING = 4


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
    enable_cueq: bool | None = False
    device: str | None = "cpu"

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
                model_path,
                enable_cueq=enable_cueq,
                device=device,
                default_dtype="float32",
            )

        return data

    @model_serializer(mode="plain")
    def _save_weights_on_serialize(self) -> dict:
        """
        When we dump this model, first save out the MACECalculator's weights
        under a directory named after model_name, then emit a dict that
        points model_path to that file, and drop the in-memory mace_calc.
        """

        if self.model_path is None:
            home = Path.home()
            outdir = home / ".cache/threedscriptors" / self.model_name
            outdir.mkdir(parents=True, exist_ok=True)

            # pick a filename (you can parameterize or version this if you like)
            model_path = str(outdir / "weights.pt")

            # assume your MACECalculator has a .save_weights(filepath) method

            torch.save(self.mace_calc.models[0], model_path)

        else:
            model_path = self.model_path

        # now emit a pure-python dict for JSON / dict dumping
        return {
            "model_name": self.model_name,
            "model_path": model_path,
            "enable_cueq": self.enable_cueq,
            "device": self.device,
            # setting this to None ensures create_default_mace_calc will reload
            "mace_calc": None,
        }


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    N_molecules: int | None
    dataset_type: DatasetTypes
    BFGS_tol: float
    BFGS_max_steps: int
    N_conformers: int = 1
    embedding_model_config: MaceCalculatorConfig | None = None
    max_atoms: int | None = None
    regression_is_normalized: bool = False
    tasks: Sequence[TaskConfig] | None = None
