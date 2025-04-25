import numpy as np

from threedscriptors.data_handling.dataset import SimilarityScreeningDataset
from threedscriptors.evaluation.descriptor_calculators import DescriptorCalculator


class SimilarityScreeningTask:
    def __init__(
        self,
        descriptor_calculator: DescriptorCalculator,
        similarity_screening_dataset: SimilarityScreeningDataset,
    ):
        self.descriptor_calculator = descriptor_calculator
        self.dataset = similarity_screening_dataset

        self.random_generator = np.random.default_rng(seed=42)

    def get_target_class_data(self, target_class_id: int):
        target_class_indices = np.argwhere(
            self.dataset.target_class_labels == target_class_id
        )

        return self.dataset[target_class_indices]

    def draw_random_active_from_class(self, class_label, num):
        active_indices = np.argwhere(
            (self.dataset.activity_decoy_labels == 1)
            & (self.dataset.target_class_labels == class_label)
        )
        indices = self.random_generator.choice(active_indices, size=num, replace=False)

        return indices

    def get_class_indices(self, class_label):
        return np.argwhere(self.dataset.target_class_labels == class_label)

    def evaluate(self, actives_resampling_frequency):
        # Calculate all molecular_descriptors
        molecular_descriptors = self.descriptor_calculator.calculate_descriptors(
            self.dataset
        )

        print(molecular_descriptors[:10, :10])
        target_classes = np.unique(self.dataset.target_class_labels)

        for class_label in target_classes:
            # Randomly drawn actives
            ref_indices = self.draw_random_active_from_class(
                class_label, num=actives_resampling_frequency
            )
            class_indices = self.get_class_indices(class_label)

            for ref_idx in ref_indices:
                reference_descriptor = molecular_descriptors[ref_idx].squeeze()

                # calculate the similarity of the entire class dataset with

                class_indices_wo_reference = class_indices[
                    np.where(class_indices != ref_idx)
                ]  # Drops the reference molecule, because it shouldnt be included in the similarity search.

                similarties = self.descriptor_calculator.get_all_similiarities(
                    reference_descriptor,
                    molecular_descriptors[class_indices_wo_reference],
                )

                activity_labels = self.dataset.activity_decoy_labels[
                    class_indices_wo_reference
                ]

                # rank the similarities
                sorting_indices = np.flip(
                    np.argsort(similarties)
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
