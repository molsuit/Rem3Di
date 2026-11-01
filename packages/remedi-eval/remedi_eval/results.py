"""The matplotlib figure artifact; the other artifacts live in ``remedi_data.runs.results``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import Field

from remedi_data.runs.results import EvalResult


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
