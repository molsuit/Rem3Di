from __future__ import annotations

import gzip
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Base class for serializable evaluation artifacts.

    ``serialize_to`` persists the artifact under ``directory`` and returns a
    small manifest entry (``{"result_type", "file_name"}``) so the runner can
    assemble a ``status.yaml`` listing every artifact it wrote without having to
    re-stat the directory.

    Data-bearing subclasses (:class:`TableResult`, :class:`ArrayResult`) also
    implement ``load`` so artifacts round-trip from disk — this is what lets the
    decoupled plotter registry re-render figures offline without re-running the
    eval. Figure / chemiscope artifacts are terminal (no ``load``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_type: str
    file_name: Path

    @abstractmethod
    def serialize_to(self, directory: Path) -> dict[str, Any]:
        """Persist the artifact under ``directory`` and return a manifest entry."""
        raise NotImplementedError

    def _manifest_entry(self) -> dict[str, Any]:
        return {"result_type": self.result_type, "file_name": str(self.file_name)}


class FigureResult(EvalResult):
    result_type: Literal["figure"] = "figure"
    figure: Figure
    save_kwargs: dict[str, Any] = Field(default_factory=dict)

    def serialize_to(self, directory: Path) -> dict[str, Any]:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output_path, **self.save_kwargs)
        plt.close(self.figure)
        return self._manifest_entry()


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
        return self._manifest_entry()


class PydanticResult(EvalResult):
    result_type: Literal["pydantic"] = "pydantic"
    obj: BaseModel

    def serialize_to(self, directory: Path) -> dict[str, Any]:
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
        return self._manifest_entry()


class TableResult(EvalResult):
    """A tabular artifact (e.g. benchmark result rows) persisted as CSV."""

    result_type: Literal["table"] = "table"
    frame: pd.DataFrame

    def serialize_to(self, directory: Path) -> dict[str, Any]:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(output_path, index=False)
        return self._manifest_entry()

    @classmethod
    def load(cls, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


class ArrayResult(EvalResult):
    """One or more named numpy arrays persisted as a compressed ``.npz``.

    Used for plot-input data (e.g. projection coordinates + colors, similarity
    distributions) so the plotter registry can re-render from disk.
    """

    result_type: Literal["array"] = "array"
    arrays: dict[str, np.ndarray]
    compressed: bool = True

    def serialize_to(self, directory: Path) -> dict[str, Any]:
        output_path = directory / self.file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # ty matches `**dict` against savez's named `allow_pickle: bool` param —
        # a false positive; the values are all ndarrays.
        if self.compressed:
            np.savez_compressed(output_path, **self.arrays)  # ty: ignore[invalid-argument-type]
        else:
            np.savez(output_path, **self.arrays)  # ty: ignore[invalid-argument-type]
        return self._manifest_entry()

    @classmethod
    def load(cls, path: Path) -> dict[str, np.ndarray]:
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}
