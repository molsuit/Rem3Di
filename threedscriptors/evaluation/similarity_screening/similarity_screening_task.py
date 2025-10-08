
class SimilarityScreeningTask(BaseEvalTask):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

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
