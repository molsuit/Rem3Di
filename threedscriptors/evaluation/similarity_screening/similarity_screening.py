from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.classification import AUROC, BinaryROC
from tqdm import tqdm

from threedscriptors.evaluation.descriptor_calculators import DescriptorCalculator


class SimilarityMetrics(Enum):
    AUROC = "auroc"
    BEDROC = "bedroc"
    ENRICHMENT_FACTOR = "enrichment_factor"
    ROC = "roc"


@dataclass
class SimilarityScreeningClassificationMetric:
    class_index: int
    sampled_metrics: list[float] | list[dict[str, list[float]]]
    metric: SimilarityMetrics
    avg_metric: float | None


@dataclass
class SimilarityScreeningRankedResults:
    ranked_similarities: np.ndarray
    ranked_activity_labels: np.ndarray
    ranked_mol_indices: np.ndarray


class SimilarityScreening:
    def __init__(
        self,
        descriptor_calculator: DescriptorCalculator,
        similarity_screening_dataset,
    ):
        self.descriptor_calculator = descriptor_calculator
        self.dataset = similarity_screening_dataset

        self.random_generator = np.random.default_rng(seed=42)

        self.results: list[SimilarityScreeningClassificationMetric] = []

    def get_target_class_data(self, target_class_id: int):
        target_class_indices = np.argwhere(
            self.dataset.target_class_labels == target_class_id
        )

        return self.dataset[target_class_indices]

    def draw_random_active_from_class(self, class_label, num):
        active_indices = np.argwhere(
            (self.dataset.active_decoy_labels == 1)
            & (self.dataset.target_class_labels == class_label)
        )
        assert active_indices.shape[0] >= num

        indices = self.random_generator.choice(
            active_indices, size=num, replace=False)

        return indices

    def get_class_indices(self, class_label):
        return np.argwhere(self.dataset.target_class_labels == class_label)

    def evaluate(self, actives_resampling_frequency):
        # Calculate all molecular_descriptors
        molecular_descriptors = self.descriptor_calculator.calculate_descriptors(
            self.dataset
        )

        target_classes = np.unique(self.dataset.target_class_labels)

        self.result_dict = {}

        for class_label in tqdm(target_classes, desc="Activity Classes", position=0):
            # Randomly drawn
            try:
                ref_indices = self.draw_random_active_from_class(
                    class_label, num=actives_resampling_frequency
                )
            except AssertionError:
                print(f"No active for class {class_label}")
                continue

            class_indices = self.get_class_indices(class_label)

            ranked_activity_labels = np.zeros(
                shape=(actives_resampling_frequency,
                       class_indices.shape[0] - 1)
                # Subtract one due to the reference molecule being removed - we dont want the highest ranked molecule to be the reference molecule with itself.
            )

            ranked_similarities = np.zeros(
                shape=(actives_resampling_frequency,
                       class_indices.shape[0] - 1)
            )
            ranked_indices = np.zeros(
                shape=(actives_resampling_frequency,
                       class_indices.shape[0] - 1)
            )

            for resampling_index, ref_idx in tqdm(
                enumerate(ref_indices),
                desc="Reference Resampling",
                position=1,
                leave=True,
            ):
                reference_descriptor = molecular_descriptors[ref_idx].squeeze()

                # calculate the similarity of the entire class dataset with

                class_indices_wo_reference = class_indices[
                    np.where(class_indices != ref_idx)
                    # Drops the reference molecule, because it shouldnt be included in the similarity search.
                ]

                similarities = self.descriptor_calculator.get_all_similiarities(
                    reference_descriptor,
                    molecular_descriptors[class_indices_wo_reference],
                )

                activity_labels = self.dataset.active_decoy_labels[
                    class_indices_wo_reference
                ]

                # rank the similarities
                sorting_indices = np.flip(
                    np.argsort(similarities)
                )  # finds the indices that sort the similarities from highest to lowest

                ranked_similarities[resampling_index,
                                    :] = similarities[sorting_indices]
                ranked_activity_labels[resampling_index, :] = activity_labels[
                    sorting_indices
                ]
                ranked_indices[resampling_index, :] = class_indices_wo_reference[
                    sorting_indices
                ]  # These are the mol ids, ranked, that correspond to the activity class under investigation

            self.result_dict[class_label] = SimilarityScreeningRankedResults(
                ranked_similarities=ranked_similarities,
                ranked_activity_labels=ranked_activity_labels,
                ranked_mol_indices=ranked_indices,
            )

        return self.result_dict

    def compute_metrics(self, enrichment_factor_percentage=0.01):
        self.compute_AUC_ROC()
        self.compute_enrichment_factor(
            subset_percentage=enrichment_factor_percentage)
        #self.compute_BEDROC()
        self.compute_roc_curve()

    def compute_AUC_ROC(self):
        auroc_fn = AUROC("binary")

        results = []
        for class_label in self.result_dict.keys():
            class_results: SimilarityScreeningRankedResults = self.result_dict[
                class_label
            ]
            number_of_resamples = class_results.ranked_similarities.shape[0]
            auroc_values = []
            for resampling_index in range(number_of_resamples):
                predictions = class_results.ranked_similarities[resampling_index, :]

                targets = class_results.ranked_activity_labels[resampling_index, :]
                auroc_values.append(
                    auroc_fn(torch.Tensor(predictions), torch.Tensor(targets))
                    .cpu()
                    .numpy()
                    .item()
                )

            results.append(
                SimilarityScreeningClassificationMetric(
                    class_index=class_label,
                    sampled_metrics=auroc_values,
                    metric=SimilarityMetrics.AUROC,
                    avg_metric=np.mean(np.array(auroc_values)).item(),
                )
            )



        self.results.extend(results)

    def compute_enrichment_factor(self, subset_percentage: float):
        # Subset percentage given as 0.01 (=top 1% of the dataset)
        ef_results = []

        for class_label in self.result_dict.keys():
            class_results: SimilarityScreeningRankedResults = self.result_dict[
                class_label
            ]

            total_number_of_compounds = class_results.ranked_similarities.shape[-1]
            subset_size = floor(subset_percentage * total_number_of_compounds)

            total_number_of_actives = np.count_nonzero(
                class_results.ranked_activity_labels[0, :]
            )

            number_of_resamples = class_results.ranked_similarities.shape[0]
            enrichment_factor_values = []
            for resampling_index in range(number_of_resamples):
                actives_in_subset = np.count_nonzero(
                    np.array(class_results.ranked_activity_labels)[
                        resampling_index, :subset_size
                    ]
                )

                ef = (actives_in_subset / subset_size) / (
                    total_number_of_actives / total_number_of_compounds
                )

                enrichment_factor_values.append(ef)

            ef_results.append(
                SimilarityScreeningClassificationMetric(
                    metric=SimilarityMetrics.ENRICHMENT_FACTOR,
                    sampled_metrics=enrichment_factor_values,
                    class_index=class_label,
                    avg_metric=np.mean(
                        np.array(enrichment_factor_values)).item(),
                )
            )

        self.results.extend(ef_results)

    def compute_roc_curve(self, N_thresholds: int = 50 ):
        roc = BinaryROC(thresholds=N_thresholds)
        results = []

        for class_label in self.result_dict.keys():
            class_results: SimilarityScreeningRankedResults = self.result_dict[
                class_label
            ]
            number_of_resamples = class_results.ranked_similarities.shape[0]

            curves = []
            for resampling_index in range(number_of_resamples):
                predictions = torch.Tensor(class_results.ranked_similarities[resampling_index, :])

                targets = torch.Tensor(class_results.ranked_activity_labels[resampling_index, :]).int()

                false_positive_rate, true_positive_rate, thresholds = roc(
                    predictions, targets
                )

                curves.append({
                    "true_positive_rate": true_positive_rate.detach()
                    .cpu()
                    .numpy(),
                    "false_positive_rate": false_positive_rate.detach()
                    .cpu()
                    .numpy(),
                    "thresholds": thresholds.detach().cpu().numpy()
                })

            tprs = [curve["true_positive_rate"] for curve in curves]
            tpr_mean = np.mean(np.array(tprs), axis=0)

            fprs = [curve["false_positive_rate"] for curve in curves]
            fpr_mean = np.mean(np.array(fprs), axis=0)

            assert tpr_mean.shape == (N_thresholds,)

            results.append(
                SimilarityScreeningClassificationMetric(
                    class_index=class_label,
                    sampled_metrics=curves,
                    metric=SimilarityMetrics.ROC,
                    avg_metric={"true_positive_rate": tpr_mean,
                                "false_positive_rate": fpr_mean, "thresholds": thresholds},
                )
            )

        self.results.extend(results)

    def compute_BEDROC(self):
        pass

    def get_similarity_metric_results(self, metric: SimilarityMetrics):
        class_labels = []
        avg_metric_values = []
        for result in self.results:
            if result.metric == metric:
                class_labels.append(result.class_index)
                avg_metric_values.append(result.avg_metric)

        return np.array(class_labels), np.array(avg_metric_values)


def plot_reference_vs_model_classification_metric(
    threedscriptor_screening: SimilarityScreening,
    reference_screening: SimilarityScreening,
    metric: SimilarityMetrics,
):
    reference_data_labels, reference_data_values = (
        reference_screening.get_similarity_metric_results(metric)
    )

    model_data_labels, model_data_values = (
        threedscriptor_screening.get_similarity_metric_results(metric)
    )

    fig = plt.figure()

    plt.plot(
        reference_data_labels,
        reference_data_values,
        label=reference_screening.descriptor_calculator.descriptor_name,
    )

    plt.plot(model_data_labels, model_data_values, label="threedscriptor")

    plt.ylabel(str(metric))
    plt.xlabel("Activity Class Index")
    plt.xticks(reference_data_labels)
    # plt.xlim(left = 0, right = reference_data_labels[-1])

    return fig


def plot_roc(threedscriptor_screening: SimilarityScreening, reference_screening: SimilarityScreening):

    threedes_rocs : list[SimilarityScreeningClassificationMetric] = [results for results in threedscriptor_screening.results if
                     results.metric == SimilarityMetrics.ROC]


    reference_rocs : list[SimilarityScreeningClassificationMetric] = [results for results in reference_screening.results if results.metric == SimilarityMetrics.ROC]

    fig, axes = plt.subplots(1, len(reference_rocs))

    if not isinstance(axes, Sequence):
        axes = [axes]

    for fig_index, (model_roc, reference_roc) in enumerate(zip(threedes_rocs, reference_rocs, strict=False)):
        axes[fig_index].plot(model_roc.avg_metric["false_positive_rate"], model_roc.avg_metric["true_positive_rate"], label = "3Des Model ROC")

        axes[fig_index].plot(reference_roc.avg_metric["false_positive_rate"], reference_roc.avg_metric["true_positive_rate"], label = "Reference Des. ROC")

        axes[fig_index].plot(np.linspace(0,1,10), np.linspace(0,1,10), color = "k")

    plt.legend()

    return fig
