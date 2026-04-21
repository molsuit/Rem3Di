class ChiralPredictionTask(BaseEvalTask):
    def __init__(self, chiral_dataset):
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


class ChiralDifferencePredictionTask(BaseEvalTask):
    def __init__(self, chiral_dataset):
        super().__init__()

        self.dataset = chiral_dataset

    def run(
        self,
        model: MultiTaskRegressionModel,
        difference_prediction_model: MolecularDifferenceRegressor,
        mean,
        std,
    ):
        pred_differences = evaluate_molecule_difference_on_dataset(
            model, self.dataset, difference_prediction_model
        )

        pred_differences = (pred_differences * std) + mean
        print(pred_differences[:10])

        len(self.dataset)

        targets = self.dataset.regression_targets.reshape(-1, 2)
        labeled_differences = torch.log(targets[:, 0]) - torch.log(targets[:, 1])

        print(labeled_differences[:10])

        model_loss = (labeled_differences - pred_differences).abs().mean()

        mean_loss = (labeled_differences).abs().mean()

        print(f"Mean Predicted Loss {mean_loss}, Model Loss = {model_loss}")

    def plot():
        pass
