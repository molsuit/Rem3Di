from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from e3nn.o3 import Irreps
from pydantic import BaseModel, PrivateAttr

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

    _raw_model: Any | None = PrivateAttr(default=None)
    _irrep_signature: Irreps | None = PrivateAttr(default=None)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    def _load_raw_model(self):
        if self._raw_model is not None:
            return self._raw_model

        from mace.calculators import MACECalculator
        from mace.calculators.foundations_models import mace_off

        if self.model_path is None:
            raw = mace_off(
                model="medium",
                device=self.device,
                default_dtype=self.dtype,
                return_raw_model=True,
                enable_cueq=False,
            )
        else:
            calc = MACECalculator(
                model_paths=str(self.model_path),
                device=self.device,
                default_dtype=self.dtype,
                enable_cueq=False,
            )
            raw = calc.models[0]

        self._raw_model = raw.to(self.torch_dtype)
        return self._raw_model

    def get_irrep_signature(self) -> Irreps:
        """Return the product-stack irrep signature of the configured MACE model."""
        if self._irrep_signature is None:
            from threedscriptors.utils.model_utils import get_mace_model_irrep_signature

            self._irrep_signature = get_mace_model_irrep_signature(
                self._load_raw_model()
            )
        return self._irrep_signature

    def get_per_layer_irreps(self) -> list[Irreps]:
        """Return the irreps emitted by each MACE message-passing layer (product
        stack) in order. The full product-stack signature is the concatenation of
        these per-layer irreps; per-layer access is needed by downstream consumers
        that want to operate on a subset of layers (e.g. only the first or second
        message passing layer).
        """
        raw = self._load_raw_model()
        return [
            Irreps(str(p.linear.__dict__["irreps_out"])) for p in raw.products
        ]

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
        from torch_sim.models.mace import MaceModel

        return MaceModel(
            model=self._load_raw_model(),
            device=torch.device(self.device),
            dtype=self.torch_dtype,
            compute_forces=self.compute_forces,
            compute_stress=self.compute_stress,
            enable_cueq=self.enable_cueq,
        )
