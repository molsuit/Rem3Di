from torch import nn

from remedi.model.model_output import ModelOutput
from remedi.model.pair_encoder import TransformerPairEncoder
from remedi.model.preprocessing.preprocessing import Preprocessor
from remedi.training.data.sample import Sample


class REM3DIModel(nn.Module):
    # A small wrapper to move components around together

    def __init__(self, preprocessor: Preprocessor, encoder: TransformerPairEncoder):
        super().__init__()

        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, sample: Sample) -> ModelOutput:
        preprocessed_sample = self.preprocessor(sample)
        molecular_descriptor = self.encoder(preprocessed_sample)

        return ModelOutput(molecular_descriptor=molecular_descriptor)
