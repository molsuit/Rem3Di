from torch import nn
import torch
from threedscriptors.data_handling.sample import Sample
from threedscriptors.model.preprocessing.preprocessing import Preprocessor

from threedscriptors.model.encoder import TransformerEncoder
from threedscriptors.model.pair_encoder import TransformerPairEncoder


# A small wrapper to move components around together


class REM3DIModel(nn.Module):

    def __init__(
        self,
        preprocessor: Preprocessor,
        encoder: TransformerEncoder | TransformerPairEncoder,
    ):

        super().__init__()

        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, sample: Sample) -> torch.Tensor:

        preprocessed_sample = self.preprocessor(sample)
        molecular_descriptor = self.encoder(preprocessed_sample)
        return molecular_descriptor
