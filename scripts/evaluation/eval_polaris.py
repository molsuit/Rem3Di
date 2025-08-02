from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.evaluation_pipeline import (
    EvalPipelineRunner,
    RegressionTestTask,
)
from threedscriptors.model.model_builder import ModelBuilder

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/169-2025_08_01_16_14_31-SingleTaskAdmetlogdPretrained"


#model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/161-2025_08_01_11_32_06-AVAdmetGeomPretraining_batch128"
model = ModelBuilder.from_directory(model_directory).build_model()



test_dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_test_full"
)

out_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/polaris_pretrain"

dataset = reload_dataset_pipeline(test_dataset_directory).build()

reg = RegressionTestTask(dataset, polaris_eval_style= True)
runner = EvalPipelineRunner(
    tasks=[reg], dataset_name="av_admet", dataset_split=DatasetSplit.TRAIN
)


runner.evaluate(model)
runner.output_results(
    output_directory=f"{out_dir}/test",
    model_name="Pretrained_regressionmodel",
)
