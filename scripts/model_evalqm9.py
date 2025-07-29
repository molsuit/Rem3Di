from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.evaluation.evaluation_pipeline import (
    EvalPipelineRunner,
    RegressionTestTask,
)
import os 
from threedscriptors.configuration.data_config import DatasetSplit

dataset_name = "qm9_test"
# load model
model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/97-2025_07_28_15_21_43-QM9_frompretrained_PCQM"

evaluation_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/"


os.makedirs(evaluation_dir + dataset_name, exist_ok= True)

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


model = ModelBuilder.from_directory(model_directory).build_model()


runner.evaluate(model)


runner.output_results(output_directory= evaluation_dir + dataset_name + "from_scratch", model_name=dataset_name)