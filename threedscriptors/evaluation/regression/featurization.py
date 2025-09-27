from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from molfeat.trans.fp import FPVecTransformer

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    cosine_similarity,
    tanimoto_similarity,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.model.regression_models import MultiTaskRegressionModel


class DescriptorCalculator(ABC):
    @abstractmethod
    def calculate_descriptors(self, dataset):
        pass

    @abstractmethod
    def calculate_similarity(self, des0, des1):
        pass

    def get_all_similiarities(
        self, reference_descriptor: np.ndarray, class_descriptors
    ):
        similiarities = np.zeros(shape=class_descriptors.shape[0])

        # vectorize the similiarities calculation
        for idx, desc in enumerate(class_descriptors):
            similiarities[idx] = self.calculate_similarity(reference_descriptor, desc)
        return similiarities


class MolfeatDescriptorCalculator(DescriptorCalculator):
    def __init__(self, descriptor_name):
        super().__init__()
        self.descriptor_name = descriptor_name
        self.featurizer = FPVecTransformer(kind=self.descriptor_name)

    def calculate_descriptors(self, dataset: MoleculeDataset):
        return self.featurizer(dataset.get_smiles_per_structure())

    def calculate_similarity(self, des0, des1):
        return tanimoto_similarity(des0, des1)


class ThreedescriptorCalculator(DescriptorCalculator):
    def __init__(
        self,
        threedescriptor_model,
        similarity_fn: Callable = cosine_similarity,
    ):
        super().__init__()

        self.model = threedescriptor_model.eval()
        self.similarity_fn = similarity_fn
        self.descriptor_name = "threedscriptor"

    def calculate_descriptors(self, dataset: MoleculeDataset):

        descriptors = evaluate_molecular_descriptor_on_dataset(self.model, dataset)

        descriptors = descriptors.numpy()

        return descriptors

    def calculate_similarity(self, des0, des1):
        return self.similarity_fn(des0, des1)
