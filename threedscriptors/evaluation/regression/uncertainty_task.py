

class RegressionUncertaintyTask(BaseEvalTask):
    def __init__(self, dataset):
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



