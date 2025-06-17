from dataclasses import dataclass
from typing import Optional
from torch import Tensor

@dataclass
class ModelOutput():
    molecular_descriptor: Tensor
    regression_predictions: Optional[Tensor] = None
    updated_pair_encoding: Optional[Tensor] = None