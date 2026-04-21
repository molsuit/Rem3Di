import gzip
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):  # np.float32, np.int64, etc.
        return x.item()
    if isinstance(x, Mapping):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, Sequence) and not isinstance(x, str | bytes):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


class EvalResult(BaseModel, ABC):
    """Base class for serializable evaluation artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_type: str
    file_name: Path

    @abstractmethod
    def serialize_to(self, directory: Path) -> dict[str, Any]:
        """Persist the artifact under ``directory`` and return a manifest entry."""
        raise NotImplementedError


class FigureResult(EvalResult):
    result_type: Literal["figure"] = "figure"
    figure: Figure
    save_kwargs: dict[str, Any] = Field(default_factory=dict)

    def serialize_to(self, directory: Path) -> dict[str, Any]:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output_path, **self.save_kwargs)
        plt.close(self.figure)


class ChemiscopeResult(EvalResult):
    result_type: Literal["chemiscope"] = "chemiscope"
    data: dict[str, Any]
    compresslevel: int = 9

    def serialize_to(self, directory: Path) -> dict[str, Any]:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if ".gz" in output_path.suffixes:
            with gzip.open(
                output_path,
                mode="wt",
                encoding="utf-8",
                compresslevel=self.compresslevel,
            ) as file:
                json.dump(self.data, file)
        else:
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(self.data, file, indent=2)


class PydanticResult(EvalResult):
    result_type: Literal["pydantic"] = "pydantic"
    obj: BaseModel

    def serialize_to(self, directory: Path):
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.obj.model_dump(mode="json")  # apply serializers → pure python types
        yaml_text = yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        output_path.write_text(yaml_text, encoding="utf-8")
