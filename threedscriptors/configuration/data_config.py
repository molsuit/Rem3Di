from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Optional

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

from threedscriptors.data_handling.dataset import (
    AtomicEmbeddingDataset,
    AtomicEmbeddingWithPositionsDataset,
    BaseDataset,
    RegressionDataset,
    RegressionDatasetwithPositions,
    RegressionWithAuxAndPositionsDataset,
    RegressionWithAuxDataset,
    SimilarityScreeningDataset,
    RegressionDatasetwithRandomWalks
)


class DatasetTypes(Enum):
    ATOMICEMBEDDING_DATASET = AtomicEmbeddingDataset
    REGRESSION_DATASET = RegressionDataset
    REGRESSION_WITH_AUX_DATASET = RegressionWithAuxDataset
    REGRESSION_WITH_POSITIONS_DATASET = RegressionDatasetwithPositions
    SIMILARITY_SCREENING_DATASET = SimilarityScreeningDataset
    ATOMICEMBEDDING_WITHPOSITIONS_DATASET = AtomicEmbeddingWithPositionsDataset
    REGRESSION_WITH_AUX_AND_POS = RegressionWithAuxAndPositionsDataset
    REGRESSION_DATATSET_WITH_RANDOMWALK = RegressionDatasetwithRandomWalks

    @classmethod
    def _missing_(cls, value: object) -> "DatasetTypes":
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value.lower():
                    return member
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")



class LabelScalingType(Enum):
    LOG = "log"


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
        
        raise TypeError("`dataset_type` must be a DatasetTypes, a BaseDataset subclass, or a registered name")

    @field_serializer("scaling")
    def _serialize_dataset_type(self, v: LabelScalingType, info):
        # turn DatasetTypes.atomic → "atomic"

        if v is None:
            return None
        
        return v.name

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


class DatasetSplit(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2



class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    N_molecules: int | None
    dataset_type: DatasetTypes
    BFGS_tol: float
    BFGS_max_steps: int
    N_conformers: int = 1
    embedding_model_config: MaceCalculatorConfig | None = None
    tasks: Sequence[TaskConfig] | None = None
    only_heavy_atoms: bool = False
    dataset_name: Optional[str] = None
    dataset_split: Optional[DatasetSplit] = None


    def get_task_names(self):
        return [tc.task_name for tc in self.tasks]

    def get_mean_std_per_task(self):

        mean = {tc.task_name : tc.mean for tc in self.tasks}
        std =  {tc.task_name : tc.std for tc in self.tasks}

        assert None not in set(mean.values())
        assert None not in set(std.values())


        return mean, std

    @field_validator("dataset_type", mode="before")
    @classmethod
    def _coerce_dataset_type(cls, v):

        if isinstance(v, DatasetTypes):
            return v

        if isinstance(v, type) and issubclass(v, BaseDataset):
            return DatasetTypes(v)
        # string → enum by name (or via _missing_)
        if isinstance(v, str):
            return DatasetTypes(v)
        raise TypeError("`dataset_type` must be a DatasetTypes, a BaseDataset subclass, or a registered name")

    @field_serializer("dataset_type")
    def _serialize_dataset_type(self, v: DatasetTypes, info):
        # turn DatasetTypes.atomic → "atomic"
        return v.name


    @field_validator('dataset_split', mode='before')
    @classmethod
    def _validate_dataset_split(cls, v):
        # allow None
        if v is None:
            return None

        # already an enum
        if isinstance(v, DatasetSplit):
            return v

        # from string like "train", "VALIDATION", etc.
        if isinstance(v, str):
            try:
                return DatasetSplit[v.strip().upper()]
            except KeyError:
                raise ValueError(
                    f"string value '{v}' is not a valid DatasetSplit; "
                    f"expected one of {[e.name for e in DatasetSplit]}"
                )

        # from integer like 0, 1, 2
        if isinstance(v, int):
            try:
                return DatasetSplit(v)
            except ValueError:
                raise ValueError(
                    f"integer value {v} is not a valid DatasetSplit; "
                    f"expected one of {[e.value for e in DatasetSplit]}"
                )

        # anything else is invalid
        raise TypeError(
            f"cannot interpret {v!r} as a DatasetSplit; "
            f"must be None, one of {[(e.name, e.value) for e in DatasetSplit]}, "
            f"or their names/values"
        )
    
    @field_serializer('dataset_split')
    def _serialize_dataset_split(self, v: Optional[DatasetSplit], _info):
        """
        Convert the enum back to a JSON-friendly form.
        Here we output the lowercase name (e.g. "train", "validation", "test"),
        but you could return v.value if you prefer integers.
        """
        if v is None:
            return None
        return v.name.lower()