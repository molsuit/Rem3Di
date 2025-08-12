import os

from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.evaluation_pipeline import (
    EvalPipelineRunner,
    RegressionTestTask,
)
from threedscriptors.model.model_builder import ModelBuilder

dataset_name = "qm9_test"
# load model

model_dirs = {
    "pretrained": "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/212-2025_08_12_17_10_43-QM9fromPCQM",
    "scratch": "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/211-2025_08_12_16_49_11-QM9fromscratch",
}


evaluation_dir = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/"
)

os.makedirs(evaluation_dir + dataset_name, exist_ok=True)


# load datasets
dataset_directory = (
    f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{dataset_name}"
)


dataset = reload_dataset_pipeline(dataset_directory).build()

tasks = [
    RegressionTestTask(dataset),
]

runner = EvalPipelineRunner(
    tasks=tasks,
    dataset_name=dataset_name,
    dataset_split=DatasetSplit.TEST,
)


for model_name, model_directory in model_dirs.items():

    model = ModelBuilder.from_directory(model_directory).build_model()

    runner.evaluate(model)

    runner.output_results(
        output_directory=evaluation_dir + dataset_name + model_name,
        model_name=dataset_name,
    )
