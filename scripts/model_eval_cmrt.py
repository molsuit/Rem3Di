from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.evaluation.evaluation_pipeline import (
    EvalPipelineRunner,
    ChiralPredictionTask,
)
import os 
from threedscriptors.configuration.data_config import DatasetSplit

dataset_name = "cmrt_test"
# load model
model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/130-2025_07_29_18_21_25-CMRT_paired_train_diffin_lin"

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