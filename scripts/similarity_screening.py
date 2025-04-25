from threedscriptors.data_handling.dataset import SimilarityScreeningDataset
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.evaluation.descriptor_calculators import (
    MolfeatDescriptorCalculator,
    ThreedescriptorCalculator,
)
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    cosine_similarity,
)
from threedscriptors.evaluation.similarity_screening import SimilarityScreeningTask
from threedscriptors.model.model_builder import ModelBuilder

directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/virtual_screening"

dataset = load_data_from_disk(directory, SimilarityScreeningDataset)
dataset.embeddings = dataset.embeddings.float()
dataset.padding_mask = dataset.padding_mask.float()
print(dataset.activity_decoy_labels)
print(dataset.target_class_labels)

ecfp = MolfeatDescriptorCalculator(descriptor_name="ecfp")
ecfp_screening = SimilarityScreeningTask(ecfp, dataset)
_, ranked_activity_labels, _ = ecfp_screening.evaluate(actives_resampling_frequency=1)
print(ranked_activity_labels[:100])


model_directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/eval_runs/similarity_screening"

mb = ModelBuilder.from_directory(model_directory)
threedescriptor_model = mb.build_model().float()

threedscriptor = ThreedescriptorCalculator(
    threedescriptor_model, similarity_fn=cosine_similarity
)
threedscriptor_screening = SimilarityScreeningTask(threedscriptor, dataset)
ranked_smiliarities, ranked_activity_labels, _ = threedscriptor_screening.evaluate(
    actives_resampling_frequency=1
)
print(ranked_activity_labels[:100])
print(ranked_smiliarities[:100])
