
import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator
from torch import nn

from threedscriptors.data_handling.sample import Sample
from threedscriptors.model.encoder import TransformerEncoder
from threedscriptors.model.model_output import ModelOutput
from threedscriptors.model.pair_encoder import TransformerPairEncoder
from threedscriptors.model.preprocessing.preprocessing import Preprocessor
from threedscriptors.utils.model_utils import get_mace_calculator_irrep_signature


class REM3DIModel(nn.Module):
    # A small wrapper to move components around together

    def __init__(
        self,
        preprocessor: Preprocessor,
        encoder: TransformerEncoder | TransformerPairEncoder,
        mace_calculator: MACECalculator | None = None
    ):

        super().__init__()

        self.preprocessor = preprocessor
        self.encoder = encoder

        self.mace_calculator = mace_calculator

        if mace_calculator is not None:
            assert get_mace_calculator_irrep_signature(mace_calculator) == preprocessor.atomic_preprocessor.config.input_irreps

    def forward(self, sample: Sample) -> torch.Tensor:

        preprocessed_sample = self.preprocessor(sample)
        molecular_descriptor = self.encoder(preprocessed_sample)

        return ModelOutput(molecular_descriptor=molecular_descriptor)



    def get_remedi_descriptor(self, atoms : Atoms):

        assert self.mace_calculator is not None

        descriptors = np.expand_dims(self.mace_calculator.get_descriptors(atoms, invariants_only=False),axis= 0)

        positions = np.expand_dims(atoms.get_positions(),axis = 0)

        padding_mask = torch.zeros((1,positions.shape[0])).bool()

        sample = Sample(embeddings = torch.from_numpy(descriptors), padding_mask= padding_mask, atomic_positions=torch.from_numpy(positions).float())



        return self.forward(sample).molecular_descriptor
