import os

from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.evaluation_pipeline import (
    ChiralPredictionTask,
    EvalPipelineRunner,
)
from threedscriptors.model.model_builder import ModelBuilder

dataset_name = "cmrt_training"
# load model
model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/171-2025_08_02_18_47_22-SingleSpeedAbsVals"

evaluation_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/"


os.makedirs(evaluation_dir + dataset_name, exist_ok= True)

# load datasets
dataset_directory = (
    f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{dataset_name}"
)


dataset = reload_dataset_pipeline(dataset_directory).build()

tasks = [
    ChiralPredictionTask(dataset),
]

runner = EvalPipelineRunner(
    tasks=tasks,
    dataset_name=dataset_name,
    dataset_split=DatasetSplit.TEST,
)


model = ModelBuilder.from_directory(model_directory).build_model()


runner.evaluate(model)
