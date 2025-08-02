from torch import nn

from threedscriptors.data_handling.sample import PreprocessedSample, Sample
from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)
from threedscriptors.model.preprocessing.geometric_preprocessor import (
    PairDistanceMatrixGeometricPreprocessor,
    RandomWalkGeometricPreprocessor,
)


class Preprocessor(nn.Module):

    def __init__(
        self,
        atomic_preprocessor,
        geometric_preprocessor: (
            RandomWalkGeometricPreprocessor
            | PairDistanceMatrixGeometricPreprocessor
            | None
        ),
    ):

        super().__init__()

        self.atomic_preprocessor : AtomicDescriptorPreprocessor = atomic_preprocessor
        self.geometric_preprocessor = geometric_preprocessor

    def forward(self, sample: Sample) -> PreprocessedSample:

        preprocessed_sample : PreprocessedSample = self.atomic_preprocessor(
            sample.embeddings, sample.padding_mask
        )



        # Now we convert the sample to a PreprocessedSample dataclass instance.
        preprocessed_sample.padding_mask = sample.padding_mask


        if self.geometric_preprocessor is not None:

            (
                preprocessed_sample.initial_pair_representation,
                preprocessed_sample.geometrical_encoding,
                preprocessed_sample.pair_mask,
            ) = self.geometric_preprocessor(sample)


        return preprocessed_sample
