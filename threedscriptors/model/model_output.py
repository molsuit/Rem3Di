from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    molecular_descriptor: Tensor | None = None
    regression_predictions: Tensor | None = None
    pair_encoding: Tensor | None = None
    pair_distances: Tensor | None = None


@dataclass
class StructuralEncodingOutput:
    rbf_encoding : Tensor | None
    pair_mask: Tensor | None = None
