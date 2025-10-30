import warnings
from pathlib import Path

import numpy as np

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.tasks import TaskType


from threedscriptors.evaluation.regression.cross_validation import (
    CrossValidationRunner,
    CVParams,
)
from threedscriptors.evaluation.regression.featurization import (
    MolfeatDescriptorCalculator, RemediDescriptorCalculator
)
from threedscriptors.evaluation.regression.learner import (
    RandomForestLearner,
    RFParams,
    RidgeLearner,
    ScalerParams,
    ScalerType,
)
from threedscriptors.model.model_builder import ModelBuilder

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net"
)


model_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/geom_drugs_350k/1-2025_10_12_18_02_05-Train"
)
eval_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/lipophilicity_benchmark_pma/cross_validation")

remedi_model = ModelBuilder.from_directory(model_dir).build_remedi_model()
dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)



desc_ecfp = MolfeatDescriptorCalculator("ecfp")
desc_remedi = RemediDescriptorCalculator(remedi_model)



X_remedi_full = desc_remedi.calculate_descriptors(dataset)
X_ecfp_full = desc_ecfp.calculate_descriptors(dataset)


for i, task_confs in enumerate(dataset.config.tasks.system_cols):

    if task_confs.task_type == TaskType.classification:
        continue

    rows = np.nonzero(dataset.mask_system[:, i])[0]
    y = dataset.targets_system.oindex[rows, i]


    X_remedi = X_remedi_full[rows, :]
    X_ecfp = X_ecfp_full[rows,:]

    cv_params = CVParams(scoring="neg_mean_absolute_error")

    remedi_ridge = RidgeLearner(scaler_params = ScalerParams(scaler_type = ScalerType.NONE))

    cvr_remedi_ridge = CrossValidationRunner(cv_params=cv_params, learner = remedi_ridge)

    res_ridge = cvr_remedi_ridge.run_kfold_repeated_cross_validation(X_remedi, y)
    print(res_ridge)
    cvr_remedi_ridge.write_output(output_directory=eval_dir)


    cv_params = CVParams(scoring="neg_mean_absolute_error")
    ecfp_ridge = RidgeLearner(scaler_params = ScalerParams(scaler_type = ScalerType.NONE))

    cvr_ecfp_ridge = CrossValidationRunner(cv_params=cv_params, learner = ecfp_ridge)

    res_ecfp_ridge = cvr_ecfp_ridge.run_kfold_repeated_cross_validation(X_ecfp, y)
    print(res_ecfp_ridge)
    cvr_ecfp_ridge.write_output(output_directory=eval_dir)

