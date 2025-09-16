from threedscriptors.data_handling.dataset_io import load_data_from_disk

from threedscriptors.data_handling.dataset import AtomicEmbeddingDataset
from threedscriptors.evaluation.evaluation_pipeline import (
    EvalPipelineRunner,
    SimilarityScreeningTask,
)
from threedscriptors.model.model_builder import ModelBuilder

directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/virtual_screening"


dataset = load_data_from_disk(directory, AtomicEmbeddingDataset)
print("loaded dataset")
dataset.embeddings = dataset.embeddings.float()
dataset.padding_mask = dataset.padding_mask.float()


model_directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/eval_runs/similarity_screening"

mb = ModelBuilder.from_directory(model_directory)
threedescriptor_model = mb.build_model().float()


task = SimilarityScreeningTask(dataset)
runner = EvalPipelineRunner(tasks=[task])
runner.evaluate(threedescriptor_model)
output_dir = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/eval_runs/train_bench"
runner.visualize(output_directory=output_dir, model_name="Test Model")
