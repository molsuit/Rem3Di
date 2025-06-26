from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    molecular_descriptor: Tensor
    regression_predictions: Tensor | None = None
    updated_pair_encoding: Tensor | None = None
