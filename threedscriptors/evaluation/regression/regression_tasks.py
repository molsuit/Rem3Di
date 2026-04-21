import numpy as np


def get_synthetic_task(dataset):
    smiles = dataset.get_smiles_per_structure()
    y = np.zeros(shape=len(smiles))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        logp = Crippen.MolLogP(mol)
        y[i] = logp
    # Load data

    # featurize

    # cross validate lass RegressionHeadPCATask(BaseEvalTask):
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
        dataset,
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
