from pathlib import Path

import torch

from threedscriptors.configuration.architecture_config import (
    EncoderOnlyArchitectureConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis.clustering import UMAPCalculator
from threedscriptors.evaluation.descriptor_analysis.clustering_task import (
    DescriptorClusteringTask,
    DescriptorElementAnalysis,
)
from threedscriptors.evaluation.evaluation_pipeline import EvalPipelineRunner

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs"
)
model_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/geom_drugs_350k/1-2025_10_12_18_02_05-Train"
)
model_name = "GEOM_DRUGS"
eval_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/geom_drugs_pretraining/geom_drugs"
)

remedi_model = EncoderOnlyArchitectureConfig.from_directory(str(model_dir)).build()
remedi_model.encoder.load_state_dict(torch.load(f"{model_dir}/encoder.pth"))
remedi_model.preprocessor.atomic_preprocessor.load_state_dict(
    torch.load(f"{model_dir}/atomic_preprocessor.pth")
)
remedi_model.preprocessor.geometric_preprocessor.load_state_dict(
    torch.load(f"{model_dir}/geometric_preprocessor.pth")
)


dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)

clustering_calculator = UMAPCalculator()
clustering_task = DescriptorClusteringTask(
    dataset=dataset, clustering_calculator=clustering_calculator
)

capacity_diagnostic_task = DescriptorElementAnalysis(dataset)

eval_pipeline = EvalPipelineRunner(
    tasks=[clustering_task, capacity_diagnostic_task], dataset_name="pcqm_benchmark"
)

eval_pipeline.evaluate(remedi_model, model_name)
eval_pipeline.output_results(output_directory=eval_dir)
