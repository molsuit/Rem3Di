import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml as vanilla_yaml
from matplotlib.lines import Line2D
from torchmetrics.functional import (
    kendall_rank_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    spearman_corrcoef,
)

from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.data_handling.data_utils import (
    get_all_atom_counts,
    get_molecular_weight,
)
from threedscriptors.data_handling.dataset import (
    BaseDataset,
    RegressionDataset,
    RegressionWithAuxDataset,
    SimilarityScreeningDataset,
)
from threedscriptors.evaluation.clustering import (
    ClusteringCalculator,
    plot_reduced_dimension,
    plot_reduced_dimension_functional_group_comparison,
)
from threedscriptors.evaluation.descriptor_calculators import (
    MolfeatDescriptorCalculator,
    ThreedescriptorCalculator,
)
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    cosine_similarity_matrix,
    euclidean_similarity,
    plot_similarity_matrix,
)
from threedscriptors.evaluation.evaluation_utils import (
    average_over_conformers,
    capacity_diagnostics,
    clip_and_log_transform,
    compute_class_std,
    evaluate_molecular_descriptor_on_dataset,
    evaluate_molecule_difference_on_dataset,
    evaluate_regression_model_on_dataset,
)
from threedscriptors.evaluation.regression_analysis import (
    add_regression_head_activations_hooks,
    get_colors_for_predictions,
)
from threedscriptors.evaluation.similarity_screening import (
    SimilarityMetrics,
    SimilarityScreening,
    plot_reference_vs_model_classification_metric,
    plot_roc,
)
from threedscriptors.model.molecule_difference_regressor import (
    MolecularDifferenceRegressor,
)
from threedscriptors.model.regression_models import MultiTaskRegressionModel


class BaseEvalTask(ABC):
    @abstractmethod
    def __init__(self):
        self.results = {}
        self.figs = {}

    @abstractmethod
    def run(self, model: MultiTaskRegressionModel):
        pass

    @abstractmethod
    def plot(self, model_name: str):
        pass


class DescriptorClusteringTask(BaseEvalTask):
    "Plots the PCA results of the Molecular Descriptor"

    def __init__(
        self, dataset: BaseDataset, clustering_calculator: ClusteringCalculator
    ):
        super().__init__()
        self.dataset = dataset
        self.clustering_calculator = clustering_calculator
        self.reduced_dimensions = None

    def run(self, model: MultiTaskRegressionModel):
        # evaluate model to get descriptors

        descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)

        self.reduced_dimensions = (
            self.clustering_calculator.get_dimensionality_reduction(descriptors, k=3)
        )

    def plot(self):
        assert self.reduced_dimensions is not None
        figs = {}
        figs["Descriptor_UMAP"] = plot_reduced_dimension(self.reduced_dimensions)

        if self.dataset.regression_targets is not None:
            for i in range(self.dataset.regression_targets.shape[1]):

                mask = self.dataset.regression_masks[:, i].bool()

                task_name = self.dataset.dataset_config.tasks[i].task_name

                fig_with_regression_coloring = plot_reduced_dimension(
                    self.reduced_dimensions[mask, :],
                    color=self.dataset.regression_targets[mask, i],
                    suptitle=f"Molecular Descriptor Clustering color = {task_name} Labels",
                )

                figs[f"Descriptor_UMAP_with_regression_labels_task_{task_name}"] = (
                    fig_with_regression_coloring
                )

                fig_with_regression_coloring_preds = plot_reduced_dimension(
                    self.reduced_dimensions[mask, :],
                    color=self.dataset.regression_targets[mask, i],
                    suptitle=f"Molecular Descriptor Clustering color = {task_name} Predictions",
                )

                figs[f"Descriptor_UMAP_with_regression_predictions_{task_name}"] = (
                    fig_with_regression_coloring_preds
                )

        if self.dataset.molecules is not None:
            molecular_weights = get_molecular_weight(self.dataset.molecules)
            num_heavy_atoms = list(
                get_all_atom_counts(self.dataset.molecules, heavy_atoms_only=True)
            )

            figs["Descriptor_UMAP_with_molecular_weight"] = plot_reduced_dimension(
                self.reduced_dimensions,
                num_heavy_atoms,
                suptitle="UMAP projection c= N heavy atoms",
            )
        self.figs = figs


class DescriptorElementAnalysis(BaseEvalTask):

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)

        self.descriptor_norms = torch.norm(self.descriptors, dim=1)

        self.run_capacity_diagnostics()

    def run_capacity_diagnostics(self):

        H_tot, utilisation, dead_dims, eig, d_eff = capacity_diagnostics(
            self.descriptors
        )

        self.eig = eig
        self.results.update(
            {
                "Largest 10 Eigenvalues": eig[::-1][:10].tolist(),
                "Smallest 10 Eigenvalues": eig[:10].tolist(),
                "Effective dimension": d_eff.item(),
                "Total entropy": H_tot.item(),
                "Utilisation": utilisation.item(),
                "Dead dims": dead_dims,
            }
        )

    def plot(self):
        fig_hist = plt.figure()
        plt.hist(
            self.descriptors.reshape(-1),
            bins=np.linspace(
                torch.min(self.descriptors), torch.max(self.descriptors), 50
            ),
        )

        fig_eigval = plt.figure()
        plt.semilogy(self.eig)
        plt.xlabel("#")
        plt.ylabel("eigenvalue")

        fig_discriptor_norm = plt.figure()
        plt.hist(
            self.descriptor_norms,
            bins=np.linspace(
                torch.min(self.descriptor_norms), torch.max(self.descriptor_norms), 50
            ),
        )
        plt.title("Distribution of Descriptor L2 Norms")

        self.figs = {
            "DescriptorElementDistribution": fig_hist,
            "EigenvalueHistogram": fig_eigval,
            "DescriptorNorm": fig_discriptor_norm,
        }


class RegressionHeadPCATask(BaseEvalTask):
    "Run the PCA analysis for the activations in each head"

    def __init__(self, dataset, clustering_calculator: ClusteringCalculator):
        super().__init__()

        self.dataset = dataset
        self.clustering_calculator = clustering_calculator

    def run(self, model: MultiTaskRegressionModel):
        activations = add_regression_head_activations_hooks(model)
        self.predictions = evaluate_regression_model_on_dataset(model, self.dataset)

        for k, v in activations.items():
            activations[k] = torch.cat(v)

        self.activations_dim_reduced = {}

        for task_head_layer, activation in activations.items():
            self.activations_dim_reduced[task_head_layer] = (
                self.clustering_calculator.get_dimensionality_reduction(activation, k=2)
            )

    def plot(self):

        figs = {}

        colors = get_colors_for_predictions(self.predictions)

        for layer, dim_red_activation in self.activations_dim_reduced.items():
            fig = plt.figure()

            plt.scatter(
                dim_red_activation[:, 0], dim_red_activation[:, 1], color=colors
            )
            plt.xlabel("Reduced Dim 1")
            plt.ylabel("Reduced Dim 2")
            plt.title(layer)

            figs[layer] = fig

        self.figs = figs


class RegressionTestTask(BaseEvalTask):
    def __init__(
        self,
        dataset: RegressionDataset | RegressionWithAuxDataset,
        polaris_eval_style=False,
    ):
        super().__init__()

        self.dataset = dataset
        self.labels = self.dataset.regression_targets
        self.polaris_eval_style = polaris_eval_style

    def run(self, model):
        assert set([tc.task_name for tc in self.dataset.dataset_config.tasks]).issubset(
            set(model.multitask_heads.task_list)
            # Check that tasks in the dataset are actually present in the prediction heads.
        )

        self.predictions = evaluate_regression_model_on_dataset(model, self.dataset)

        self.standardized_predictions = evaluate_regression_model_on_dataset(
            model, self.dataset, undo_standardization=True
        )

        self.calculate_model_loss()

    def plot(self):
        figs = {}

        figs.update(self._plot_prediction_vs_reference())

        self.figs = figs

    def get_labeled_task_data(self, task_idx):

        task_mask = torch.tensor(self.dataset.regression_masks[:, task_idx], dtype=bool)

        task_predictions = self.standardized_predictions[:, task_idx]
        predictions_with_labels = task_predictions[task_mask].detach().cpu().numpy()

        task_labels = self.dataset.regression_targets[:, task_idx]
        labels = task_labels[task_mask].detach().cpu().numpy()

        mol_ids_with_labels = [
            s_id.structure_id
            for s_id, keep in zip(self.dataset.structure_ids, task_mask, strict=False)
            if keep
        ]

        return mol_ids_with_labels, predictions_with_labels, labels

    def _plot_prediction_vs_reference(self) -> dict[str : plt.Figure]:
        fig, ax = plt.subplots()

        for task_idx, task in enumerate(self.dataset.dataset_config.tasks):

            _, predictions_with_labels, labels = self.get_labeled_task_data(task_idx)
            plt.scatter(
                predictions_with_labels, labels, label=task.task_name, alpha=0.6, s=0.5
            )

        plt.xlabel("Predictions")
        plt.ylabel("Reference Labels")

        ax.legend(
            # x=1.02 means just to the right of the axes
            bbox_to_anchor=(1.07, 1),
            loc="upper left",
            borderaxespad=0,
        )
        fig.tight_layout()

        return {"RefVSPredScatter": fig}

    def calculate_model_loss(self):

        # Get the model predictions for all tasks.

        if self.dataset.dataset_config.N_conformers > 1:
            self.standardized_predictions = average_over_conformers(
                self.dataset.structure_ids, self.standardized_predictions
            )
            # Average out the mean prediction around conformers

        preds = self.standardized_predictions

        targets = self.dataset.regression_targets
        masks = self.dataset.regression_masks.bool()

        loss_results = {}

        for task_idx, task in enumerate(self.dataset.dataset_config.tasks):

            sliced_preds = preds[masks[:, task_idx].squeeze(), task_idx]

            sliced_targets = targets[masks[:, task_idx].squeeze(), task_idx]

            if self.polaris_eval_style:
                print(task.task_name)
                if "LogD" not in task.task_name:
                    print(f"clip and logging {task.task_name}")
                    sliced_preds = clip_and_log_transform(sliced_preds)
                    sliced_targets = clip_and_log_transform(sliced_targets)

            mae = mean_absolute_error(sliced_targets, sliced_preds)
            mse = mean_squared_error(sliced_preds, sliced_targets)

            r2 = r2_score(sliced_preds, sliced_targets)
            pearson_r = pearson_corrcoef(sliced_preds, sliced_targets)
            kendall_tau = kendall_rank_corrcoef(sliced_preds, sliced_targets)
            spearman_rho = spearman_corrcoef(sliced_preds, sliced_targets)

            loss_results.update(
                {
                    task.task_name: {
                        "mean absolute error": mae.item(),
                        "mean squared error": mse.item(),
                        "r2": r2.item(),
                        "spearman rho": spearman_rho.item(),
                        "pearson r": pearson_r.item(),
                        "kendall tau": kendall_tau.item(),
                    }
                }
            )

        self.results.update({"Regression Eval Stats": loss_results})


class RegressionUncertaintyTask(BaseEvalTask):
    def __init__(self, dataset: RegressionDataset | RegressionWithAuxDataset):
        super().__init__()

        self.dataset = dataset
        self.labels = self.dataset.regression_targets

    def run(self, model):
        assert set([tc.task_name for tc in self.dataset.dataset_config.tasks]).issubset(
            set(model.multitask_heads.task_list)
            # Check that tasks in the dataset are actually present in the prediction heads.
        )

        self.predictions = evaluate_regression_model_on_dataset(model, self.dataset)

        self._check_uncertainty()

    def plot(self):
        figs = {}
        figs.update(self._plot_conformer_uncertainty_histogram())

        self.figs = figs

    def _plot_conformer_uncertainty_histogram(self):

        unlabeled_std_dev = np.squeeze(
            np.concatenate(self.results["unlabeled_std_devs_conf_predictions"])
        )

        labeled_std_dev = np.squeeze(
            np.concatenate(self.results["labeld_std_devs_conf_predictions"])
        )

        fig = plt.figure(figsize=(8, 6))

        max_std_dev = max(np.max(unlabeled_std_dev), np.max(labeled_std_dev))
        min_std_dev = min(np.min(unlabeled_std_dev), np.min(labeled_std_dev))

        bins = np.logspace(np.log10(min_std_dev), np.log10(max_std_dev), 25)
        bins = np.linspace(min_std_dev, 0.5, 50)
        labeled_counts, labeled_bins = np.histogram(labeled_std_dev, bins=bins)
        unlabeled_counts, unlabeled_bins = np.histogram(unlabeled_std_dev, bins=bins)

        plt.stairs(labeled_counts, labeled_bins, label="Labelled Predictions")
        plt.stairs(unlabeled_counts, unlabeled_bins, label="Unlabelled Predictions")

        plt.ylabel("Counts")
        plt.xlabel(
            "Std deviation of Model Predictions between conformers of the same molecule (in std. units)"
        )
        # plt.xscale("log")
        # plt.yscale("log")
        plt.legend()

        return {"Conformer_std_dev": fig}

    def _check_uncertainty(self):
        "Plots the uncertainty of the regression predictions within the conformers of a single molecule"

        labeled_std_devs = []
        unlabeld_std_devs = []

        for task_idx, _ in enumerate(self.dataset.dataset_config.tasks):

            mol_ids_with_labels, predictions_with_labels, labels = (
                self.get_labeled_task_data(task_idx)
            )

            # This calculates the std deviation for the predicitions that are labeled and not masked
            _, task_conformer_std_dev_labeled = compute_class_std(
                predictions_with_labels.reshape(-1, 1), mol_ids_with_labels
            )

            mol_ids_without_labels, predictions_without_lables = (
                self.get_unlabeled_task_data(task_idx)
            )

            # This calculates the std deviation for the predicitions that are unlabeld
            _, task_conformer_std_dev_unlabeled = compute_class_std(
                predictions_without_lables.reshape(-1, 1), mol_ids_without_labels
            )

            labeled_std_devs.append(task_conformer_std_dev_labeled.tolist())
            unlabeld_std_devs.append(task_conformer_std_dev_unlabeled.tolist())

        self.results = {
            "Conformer_Uncertainty": {
                "labeld_std_devs_conf_predictions": labeled_std_devs,
                "unlabeled_std_devs_conf_predictions": unlabeld_std_devs,
            }
        }

    def get_unlabeled_task_data(self, task_idx):

        task_predictions = self.predictions[:, task_idx]

        unlabeld_task_mask = torch.logical_not(
            torch.tensor(self.dataset.regression_masks[:, task_idx], dtype=bool)
        )

        predictions_without_lables = task_predictions[unlabeld_task_mask]
        mol_ids_without_labels = [
            mol_id
            for mol_id, keep in zip(
                self.dataset.mol_ids, unlabeld_task_mask, strict=False
            )
            if keep
        ]

        return mol_ids_without_labels, predictions_without_lables

    def get_labeled_task_data(self, task_idx):

        task_mask = torch.tensor(self.dataset.regression_masks[:, task_idx], dtype=bool)

        task_predictions = self.predictions[:, task_idx]
        predictions_with_labels = task_predictions[task_mask].detach().cpu().numpy()

        task_labels = self.dataset.regression_targets[:, task_idx]
        labels = task_labels[task_mask].detach().cpu().numpy()

        mol_ids_with_labels = [
            mol_id
            for mol_id, keep in zip(self.dataset.mol_ids, task_mask, strict=False)
            if keep
        ]

        return mol_ids_with_labels, predictions_with_labels, labels


class SimilarityScreeningTask(BaseEvalTask):
    def __init__(self, dataset):
        super().__init__()

        self.dataset: SimilarityScreeningDataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        threedscriptor = ThreedescriptorCalculator(
            model, similarity_fn=euclidean_similarity
        )
        self.model_screening = SimilarityScreening(threedscriptor, self.dataset)
        self.model_screening.evaluate(actives_resampling_frequency=10)
        self.model_screening.compute_metrics(enrichment_factor_percentage=0.01)
        print(self.model_screening.results)

        ecfp = MolfeatDescriptorCalculator(descriptor_name="ecfp")
        self.reference_screening = SimilarityScreening(ecfp, self.dataset)
        self.reference_screening.evaluate(actives_resampling_frequency=10)
        self.reference_screening.compute_metrics(enrichment_factor_percentage=0.01)
        print(self.reference_screening.results)

    def plot(self):
        metrics = [SimilarityMetrics.AUROC, SimilarityMetrics.ENRICHMENT_FACTOR]
        figs = {}

        for metric in metrics:
            figs[f"SimilarityScreening{metric!s}"] = (
                plot_reference_vs_model_classification_metric(
                    threedscriptor_screening=self.model_screening,
                    reference_screening=self.reference_screening,
                    metric=metric,
                )
            )

        figs["ROCs"] = plot_roc(self.model_screening, self.reference_screening)

        self.figs = figs


class ChiralDifferencePredictionTask(BaseEvalTask):

    def __init__(self, chiral_dataset: RegressionWithAuxDataset):
        super().__init__()

        self.dataset = chiral_dataset

    def run(
        self,
        model: MultiTaskRegressionModel,
        difference_prediction_model: MolecularDifferenceRegressor,mean, std
    ):

        pred_differences = evaluate_molecule_difference_on_dataset(model,self.dataset, difference_prediction_model)

        pred_differences = (pred_differences * std) + mean
        print(pred_differences[:10])
        
        
        N_pairs = len(self.dataset)
        
        targets = self.dataset.regression_targets.reshape(-1,2)
        labeled_differences = torch.log(targets[:,0])- torch.log(targets[:,1])

        print(labeled_differences[:10])

        
        model_loss = (labeled_differences-pred_differences).abs().mean()


        mean_loss = (labeled_differences).abs().mean()

        print(f"Mean Predicted Loss {mean_loss}, Model Loss = {model_loss}")



    def plot():
        pass

class ChiralPredictionTask(BaseEvalTask):

    def __init__(self, chiral_dataset: RegressionWithAuxDataset):
        super().__init__()

        self.dataset = chiral_dataset

    def run(self, model: MultiTaskRegressionModel):

        assert "cmrt" in model.multitask_heads.task_heads.keys()

        self.predictions = evaluate_regression_model_on_dataset(
            model, self.dataset, undo_standardization=True
        )

        # predictions_per_conf_batch = self.reshape_by_enantiomers()
        self.calculate_mean_prediction_loss()

    def reshape_by_enantiomers(self):
        # conf batch = 2 enationmers of the "same" molecule.
        # Slice only the predictions for which the full conformers are available

        prediction_by_enantiomer_batch = self.predictions.view(
            -1, 2 * self.dataset.dataset_config.N_conformers
        )

        targets_by_enantiomer_batch = self.dataset.regression_targets.view(
            -1, 2 * self.dataset.dataset_config.N_conformers
        )

        molecule_ids = torch.Tensor(
            [sid.molecule_id for sid in self.dataset.structure_ids]
        ).reshape(-1, 2 * self.dataset.dataset_config.N_conformers)

        row_uniform = (molecule_ids == molecule_ids[:, [0]]).all(dim=1)

        assert (
            row_uniform.all().item()
        ), "Found a row with multiple distinct molecule_ids"

        return prediction_by_enantiomer_batch, targets_by_enantiomer_batch

    def get_enantiomer_predictions_and_mean(self):

        prediction_by_enantiomer_batch, regression_targets_by_enantiomer_batch = (
            self.reshape_by_enantiomers()
        )

        mean_predictions = regression_targets_by_enantiomer_batch.mean(
            dim=1, keepdim=True
        ).repeat(1, regression_targets_by_enantiomer_batch.size(1))

        return (
            prediction_by_enantiomer_batch,
            regression_targets_by_enantiomer_batch,
            mean_predictions,
        )

    def calculate_mean_prediction_loss(self):

        enantiomer_batched_predictions, enantiomer_batched_targets, mean_predictions = (
            self.get_enantiomer_predictions_and_mean()
        )

        print(enantiomer_batched_predictions[:10, :])
        print(enantiomer_batched_targets[:10, :])
        print(mean_predictions[:10, :])

        mean_predictions = mean_predictions.view(-1, 1)
        enantiomer_targets = enantiomer_batched_targets.view(-1, 1)
        model_predictions = enantiomer_batched_predictions.view(-1, 1)

        rmse_mean_prediction = (
            ((mean_predictions - enantiomer_targets) ** 2).mean().sqrt()
        )
        rmse_model_prediction = (
            ((model_predictions - enantiomer_targets) ** 2).mean().sqrt()
        )

        print(f"Mean Prediction of Enantiomer RMSE {rmse_mean_prediction} ")
        print(f"MOdel Prediction of Enantiomer RMSE {rmse_model_prediction}")

    def calculate_unstandardized_predictions(self):
        # Remove the log normalization and compare to the

        prediction_by_enantiomer_batch, regression_targets_by_enantiomer_batch = (
            self.reshape_by_enantiomers()
        )

        # get the mean prediction and std per batch

        mean_prediction_per_enantiomer = torch.mean(
            prediction_by_enantiomer_batch, dim=1
        )

        regression_target_per_enantiomer = torch.mean(
            regression_targets_by_enantiomer_batch, dim=1
        )

        cmrt_task = next(
            t for t in self.dataset.dataset_config.tasks if t.task_name == "cmrt"
        )

        cmrt_std = cmrt_task.std
        cmrt_mean = cmrt_task.mean

        destandardized_prediction = torch.exp(
            mean_prediction_per_enantiomer * cmrt_std + cmrt_mean
        )

        destandardized_target = torch.exp(
            regression_target_per_enantiomer * cmrt_std + cmrt_mean
        )

        rmse = mean_squared_error(
            destandardized_prediction, destandardized_target, squared=False
        )

        r2 = r2_score(destandardized_prediction, destandardized_target)

        print(f"Root mean square error  {rmse}")
        print(f"R2 coeffeicient{r2}")

    def plot(self):

        figs = {}
        figs["chiral_parity_plot"] = self._plot_reference_vs_prediction()
        figs["Distance_to_mean_hist"] = self.plot_distance_to_mean()

        figs["chiral_conformer_comparison"] = (
            self.plot_distribution_conformer_predictions()
        )
        self.figs = figs

    def _plot_reference_vs_prediction(self):
        fig = plt.figure()

        plt.scatter(self.predictions, self.dataset.regression_targets, s=1, alpha=0.5)
        plt.xlabel("Model Predictions")
        plt.ylabel("Reference Values")

        return fig

    def plot_distribution_conformer_predictions(self):

        prediction_by_enantiomer_batch, regression_targets_by_enantiomer_batch = (
            self.reshape_by_enantiomers()
        )

        fig = plt.figure()

        plot_N_molecules = 10
        n_confs_per_enantiomer = int(prediction_by_enantiomer_batch.size(1) / 2)

        x = np.ones(shape=(n_confs_per_enantiomer))

        y_min = torch.inf

        y_max = -torch.inf

        for idx, predictions, targets in zip(
            range(plot_N_molecules),
            prediction_by_enantiomer_batch,
            regression_targets_by_enantiomer_batch,
            strict=False,
        ):

            class_pos = idx * 6

            e0 = predictions[:n_confs_per_enantiomer]
            e1 = predictions[n_confs_per_enantiomer:]

            center = torch.mean(targets)

            y_min = min(
                torch.min(predictions - center), y_min, torch.min(targets - center)
            )
            y_max = max(
                torch.max(predictions - center), y_max, torch.max(targets - center)
            )

            plt.vlines(x=class_pos, ymin=-5, ymax=5, colors="k")

            plt.scatter(
                (class_pos + 2) * x, e0 - center, c="tab:blue", marker="*", label="E0PS"
            )
            plt.scatter(
                (class_pos + 3) * x,
                e1 - center,
                c="tab:orange",
                marker="*",
                label="E1PS",
            )
            plt.scatter(class_pos + 4, targets[0] - center, c="tab:blue", label="GT_E0")
            plt.scatter(
                class_pos + 4,
                targets[n_confs_per_enantiomer] - center,
                c="tab:orange",
                label="GT_E1",
            )

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="orange",
                linestyle="None",
                label="Enantiomer 0",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="blue",
                linestyle="None",
                label="Enantiomer 1",
            ),
            Line2D(
                [0], [0], marker="o", color="black", linestyle="None", label="Reference"
            ),
        ]

        plt.ylim([1.2 * y_min, 1.2 * y_max])

        plt.xlim([0, 6 * 10])

        plt.xticks(ticks=np.linspace(3, 57, 10), labels=[str(i) for i in range(10)])

        plt.yticks([])
        plt.ylabel("Retention Time [a.u]")
        plt.xlabel("Molecule")
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
        )
        return fig

    def compare_ps_no_ps(no_ps_model, ps_model):
        pass

    def plot_distance_to_mean(self):

        enantiomer_batched_predictions, enantiomer_batched_targets, mean_predictions = (
            self.get_enantiomer_predictions_and_mean()
        )

        fig = plt.figure()

        distance_from_mean = enantiomer_batched_predictions.view(
            -1, 1
        ) - mean_predictions.view(-1, 1)

        plt.hist(distance_from_mean, bins=50)

        plt.xlabel("Distance from the mean predictions")

        return fig


class DescriptorSimilarityAnalysisTask(BaseEvalTask):
    def __init__(self, dataset: BaseDataset):
        super().__init__()

        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel, normalize_descriptors=False):

        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)
        if normalize_descriptors:
            raise NotImplementedError
        self.similarity_matrix = cosine_similarity_matrix(self.descriptors)

    def plot(self):
        figs = {}

        figs["CosineSimilarityMatrixOfMolDescriptors"] = plot_similarity_matrix(
            self.similarity_matrix
        )

        self.figs = figs


class EnolThiolEvalTask(DescriptorClusteringTask):
    def __init__(self, dataset: BaseDataset, clustering_calculator):
        super().__init__(dataset=dataset, clustering_calculator=clustering_calculator)

    def plot(self) -> plt.Figure:
        fig = plot_reduced_dimension_functional_group_comparison(
            self.reduced_dimensions, self.dataset.smiles_list
        )
        return {"FunctionalGroupComparisonTask": fig}


class EvalPipelineRunner:
    def __init__(
        self, tasks: list[BaseEvalTask], dataset_name, dataset_split: DatasetSplit
    ):

        self.tasks = tasks
        self.dataset_label = f"{dataset_name}_{dataset_split.name.lower()}"

    def evaluate(self, model: MultiTaskRegressionModel):
        model.eval()

        for task in self.tasks:
            task.run(model)

    def output_results(self, output_directory: str, model_name: str):

        os.makedirs(output_directory, exist_ok=True)
        figs = self.visualize(output_directory, model_name)

        report_str = self.write_results(output_directory, model_name)

        return figs, report_str

    def visualize(self, output_directory: str, model_name: str):
        figs: dict[str : plt.Figure] = {}  # taskname : Figure

        os.makedirs(output_directory, exist_ok=True)
        for task in self.tasks:
            task.plot()
            figs.update(task.figs)

        for fig_name, fig in figs.items():
            fig.savefig(
                f"{output_directory}/{fig_name}_{model_name}_{self.dataset_label}.pdf"
            )

        return figs

    def write_results(self, output_directory, model_name):

        result_dict = {}

        for task in self.tasks:
            if any(task.results):
                result_dict.update(task.results)

        with open(
            f"{output_directory}/training_results_{model_name}_{self.dataset_label}.yaml",
            "w",
        ) as f:
            vanilla_yaml.dump(result_dict, f)

        return result_dict
