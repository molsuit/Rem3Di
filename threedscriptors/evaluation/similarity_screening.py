from collections.abc import Callable

import numpy as np

from threedscriptors.data_handling.dataset import SimilarityScreeningDataset
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    calculate_similiarities,
)
from threedscriptors.model.regression_models import MultiTaskRegressionModel
from threedscriptors.utils.descriptor_calculation import calculate_molfeat_fingerprint


def fingerprint_closure(fingerprint_algorithm):
    def calculate_fingerprint(dataset: SimilarityScreeningDataset):
        fingerprints = calculate_molfeat_fingerprint(
            dataset.smiles_list, fingerprint_name=fingerprint_algorithm
        )
        return fingerprints

    return calculate_fingerprint


def threedscriptor_calculator_closure(model: MultiTaskRegressionModel):
    def calculate_threedscriptors(dataset: SimilarityScreeningDataset):
        return model.get_molecular_descriptor(dataset.embeddings, dataset.padding_mask)

    return calculate_threedscriptors


class SimilarityScreeningTask:
    def __init__(
        self,
        descriptor_fn: Callable,
        similarity_screening_dataset: SimilarityScreeningDataset,
        descriptor_similarity_fn: Callable,
    ):
        self.calculate_molecular_descriptors = descriptor_fn
        self.dataset = similarity_screening_dataset
        self.descriptor_similarity_fn = descriptor_similarity_fn

    def get_target_class_data(self, target_class_id: int):
        target_class_indices = np.argwhere(
            self.dataset.target_class_labels == target_class_id
        )

        return self.dataset[target_class_indices]

    def evaluate(self, actives_resampling_frequency):
        # Calculate all molecular_descriptors
        molecular_descriptors = self.calculate_molecular_descriptors(self.dataset)

        for class_label in self.dataset.target_classes:
            # Randomly drawn actives
            ref_indices = self.dataset.draw_random_active_from_class(
                class_label, num=actives_resampling_frequency
            )
            class_indices = self.dataset.get_class_indices(class_label)

            for ref_idx in ref_indices:
                reference_descriptor = molecular_descriptors[ref_idx]

                # calculate the similarity of the entire class dataset with

                class_indices_wo_reference = class_indices[
                    np.where(class_indices != ref_idx)
                ]  # Drops the reference molecule, because it shouldnt be included in the similarity search.

                similarties = calculate_similiarities(
                    reference_descriptor,
                    molecular_descriptors[class_indices_wo_reference],
                    self.descriptor_similarity_fn,
                )

                activity_labels = self.dataset.activity_decoy_labels[
                    class_indices_wo_reference
                ]

                # rank the similarities
                sorting_indices = np.flip(
                    np.argsort(similarties, order="")
                )  # finds the indices that sort the similarities from highest to lowest

                ranked_similarities = similarties[sorting_indices]
                ranked_activitiy_labels = activity_labels[sorting_indices]
                ranked_indices = class_indices[sorting_indices]

        return ranked_similarities, ranked_activitiy_labels, ranked_indices

        # Threshhold metrics

        # Ranking Metrics
        # Calculate AUC, ER metrics
        # Early Enrichment
        # BEDROC

        # calculate the
