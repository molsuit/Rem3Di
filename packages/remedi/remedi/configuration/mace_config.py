from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from e3nn.o3 import Irreps
from pydantic import PrivateAttr

from remedi.configuration.config_utils import StrictBaseModel
from remedi.utils.model_utils import concatenate_irreps, mace_per_layer_irreps

if TYPE_CHECKING:
    from remedi.model.featurizer import MaceFeaturizer


class MaceConfig(StrictBaseModel):
    """The MACE model whose node features describe the atoms, and how to run it.

    `model_path=None` falls back to the MACE-OFF medium checkpoint. The raw
    model is always loaded on the CPU (reading its irreps must not need a GPU);
    `build_featurizer` moves it to `device`, so `device` may still be changed
    after validation (e.g. to run a checkpoint on another machine).
    `enable_cueq` only takes effect on CUDA devices. `compile` evaluates the
    featurizer through torch.compile (on the CPU that needs gcc >= 10).
    """

    model_path: Path | None = None
    device: str = "cuda"
    dtype: Literal["float32", "float64"] = "float64"
    enable_cueq: bool = True
    compile: bool = True

    _raw_model: Any | None = PrivateAttr(default=None)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def uses_cuequivariance(self) -> bool:
        """cuequivariance kernels are CUDA-only; on the CPU the e3nn model runs."""
        return self.enable_cueq and torch.device(self.device).type == "cuda"

    def _load_raw_model(self):
        if self._raw_model is not None:
            return self._raw_model

        if self.model_path is None:
            from mace.calculators.foundations_models import mace_off

            raw = mace_off(
                model="medium",
                device="cpu",
                default_dtype=self.dtype,
                return_raw_model=True,
                enable_cueq=False,
            )
        else:
            from mace.modules.polar.conversion import load_model_file

            # The fork's loader also reads PolarMACE files from before its
            # electrostatics moved into mace.modules.polar.
            raw = load_model_file(str(self.model_path), map_location="cpu")

        self._raw_model = raw.to(self.torch_dtype).eval()
        return self._raw_model

    def get_irrep_signature(self) -> Irreps:
        """The irreps of the full node features: every layer's, concatenated."""
        return concatenate_irreps(self.get_per_layer_irreps())

    def get_per_layer_irreps(self) -> list[Irreps]:
        """Return the irreps emitted by each MACE message-passing layer (product
        stack) in order. The full product-stack signature is the concatenation of
        these per-layer irreps; per-layer access is needed by downstream consumers
        that want to operate on a subset of layers (e.g. only the first or second
        message passing layer).
        """
        return mace_per_layer_irreps(self._load_raw_model())

    def build_featurizer(self) -> MaceFeaturizer:
        """The node-feature model (``remedi.model.featurizer``) on ``device``."""
        from remedi.model.featurizer import build_mace_featurizer

        return build_mace_featurizer(
            self._load_raw_model(),
            device=torch.device(self.device),
            dtype=self.torch_dtype,
            enable_cueq=self.uses_cuequivariance,
            use_compile=self.compile,
        )
