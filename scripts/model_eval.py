

import pydantic_yaml as pyaml

from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.dataset import RegressionWithAuxDataset
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.training_evaluation import chiral_regression_pipeline
from threedscriptors.model.model_builder import ModelBuilder

model_run = "cmrt"
dataset = "cmrt"
#load model
model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/27-2025_06_11_10_05_51-baseline_ps_generation"


model = ModelBuilder.from_directory(model_directory).build_model()

train_config = pyaml.parse_yaml_file_as(TrainingConfig, file = f"{model_directory}/training_config.yaml")


#load datasets
dataset_directory = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{dataset}"


dataset = reload_dataset_pipeline(dataset_directory, normalize_inputs= False, normalize_targets= train_config.normalized_targets, dataset_cls=RegressionWithAuxDataset).build()

# run the eval pipeline
#runner = regression_pipeline(dataset)
runner = chiral_regression_pipeline(dataset)

runner.evaluate(model)
runner.visualize(output_directory=model_directory, model_name=model_run)
