import torch
from torch import nn

from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocess,
)
from threedscriptors.model.preprocessing.geometric_preprocessor import (
    RandomWalkGeometricPreprocessor,
    PairDistanceMatrixGeometricPreprocessor,
)

from threedscriptors.data_handling.sample import Sample, PreprocessedSample


class Preprocessor(nn.Module):

    def __init__(
        self,
        atomic_preprocessor: AtomicDescriptorPreprocess,
        geometric_preprocessor: (
            RandomWalkGeometricPreprocessor
            | PairDistanceMatrixGeometricPreprocessor
            | None
        ),
    ):

        super().__init__()

        self.atomic_preprocessor = atomic_preprocessor
        self.geometric_preprocessor = geometric_preprocessor

    def forward(self, sample: Sample) -> PreprocessedSample:

        preprocessed_atomic_embeddings = self.atomic_preprocessor(
            sample.embeddings, sample.padding_mask
        )

        # Now we convert the sample to a PreprocessedSample dataclass instance.
        preprocessed_sample = PreprocessedSample(
            preprocessed_atomic_embeddings=preprocessed_atomic_embeddings,
            padding_mask=sample.padding_mask,
        )

        if self.geometric_preprocessor is not None:

            (
                preprocessed_sample.initial_pair_representation,
                preprocessed_sample.geometrical_encoding,
                preprocessed_sample.pair_mask,
            ) = self.geometric_preprocessor(sample)

        return preprocessed_sample
