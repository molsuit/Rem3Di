from enum import Enum
from pathlib import Path

import torch
from mace.calculators import MACECalculator
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)


class LabelScalingType(str, Enum):
    NONE = "none"
    Z = "z"
    LOG_Z = "log_z"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return None
        if isinstance(value, LabelScalingType):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"none", "identity"}:
                return cls.NONE
            if v in {"z", "standard", "standardize"}:
                return cls.Z
            if v in {"log", "log_z", "log-standardize", "log-standardise"}:
                return cls.LOG_Z
        # Returning None lets Pydantic raise its usual validation error
        return None



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





class TaskConfig(BaseModel):
    task_name: str
    mean: float | None = None
    std: float | None = None
    scaling: LabelScalingType | None = None
    has_auxillary_data: bool = False
    auxillary_data_dimension: int | None = None

    @field_validator("scaling", mode="before")
    @classmethod
    def _coerce_scaling(cls, v):

        if v is None:
            return None

        if isinstance(v, LabelScalingType):
            return v
        # string → enum by name (or via _missing_)
        if isinstance(v, str):
            print(v.lower())
            return LabelScalingType(v.strip().lower())

        raise TypeError(
            "`dataset_type` must be a DatasetTypes, a BaseDataset subclass, or a registered name"
        )

    @field_serializer("scaling")
    def _serialize_scaling(self, v: LabelScalingType | None, _info):
        return None if v is None else v.name



    def get_task_names(self):
        if self.tasks is None:
            return None
        else:
            return [tc.task_name for tc in self.tasks]

    def get_mean_std_per_task(self):

        mean = {tc.task_name: tc.mean for tc in self.tasks}
        std = {tc.task_name: tc.std for tc in self.tasks}

        assert None not in set(mean.values())
        assert None not in set(std.values())

        return mean, std

