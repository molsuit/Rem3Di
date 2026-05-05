from dataclasses import dataclass

from torch import Tensor

from threedscriptors.model.molecular_descriptor import MolecularDescriptor


@dataclass
class ModelOutput:
    molecular_descriptor: MolecularDescriptor | None = None
    regression_predictions: Tensor | None = None
