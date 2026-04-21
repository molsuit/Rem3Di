from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from mace.calculators.mace import MACECalculator
    from torch_sim.models.mace import MaceModel


class MaceConfig(BaseModel):
    """Configuration for loading a MACE model for both ASE and torch-sim inference.

    `model_path=None` falls back to the MACE-OFF medium checkpoint.
    """

    model_path: Path | None = None
    device: str = "cuda"
    dtype: Literal["float32", "float64"] = "float64"
    compute_forces: bool = False
    compute_stress: bool = False
    enable_cueq: bool = True

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    def build_ase_calculator(self) -> MACECalculator:
        from mace.calculators import MACECalculator
        from mace.calculators.foundations_models import mace_off

        if self.model_path is None:
            return mace_off(
                model="medium",
                device=self.device,
                default_dtype=self.dtype,
                enable_cueq=self.enable_cueq,
            )
        return MACECalculator(
            model_paths=str(self.model_path),
            device=self.device,
            default_dtype=self.dtype,
            enable_cueq=self.enable_cueq,
        )

    def build_torch_sim_model(self) -> MaceModel:
        from mace.calculators import MACECalculator
        from mace.calculators.foundations_models import mace_off
        from torch_sim.models.mace import MaceModel

        if self.model_path is None:
            raw = mace_off(
                model="medium",
                device=self.device,
                default_dtype=self.dtype,
                return_raw_model=True,
                enable_cueq=self.enable_cueq,
            )
        else:
            calc = MACECalculator(
                model_paths=str(self.model_path),
                device=self.device,
                default_dtype=self.dtype,
                enable_cueq=self.enable_cueq,
            )
            raw = calc.models[0]
        raw = raw.to(self.torch_dtype)
        return MaceModel(
            model=raw,
            device=torch.device(self.device),
            dtype=self.torch_dtype,
            compute_forces=self.compute_forces,
            compute_stress=self.compute_stress,
            enable_cueq=self.enable_cueq,
        )
