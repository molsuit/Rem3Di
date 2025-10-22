from pathlib import Path
import warnings
import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.evaluation.regression.cross_validation import (
    CrossValidationRunner,
    CVParams,
)
from threedscriptors.evaluation.regression.featurization import (
    MolfeatDescriptorCalculator,
)
from threedscriptors.evaluation.regression.learner import (
    RandomForestLearner,
    RFParams,
    RidgeLearner,
    ScalerParams,
    ScalerType,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from threedscriptors.data_handling.dataset.tasks import TaskType
from threedscriptors.model.model_builder import ModelBuilder

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net"
)
model_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/geom_drugs_350k/1-2025_10_12_18_02_05-Train"
)
eval_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/lipophilicity_benchmark_pma/cross_validation")

remedi_model = ModelBuilder.from_directory(model_dir).build_remedi_model()
dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
ds = TrainingMoleculeDataset(dataset_dir, get_item=pos_emb_getitem)
desc_ecfp = MolfeatDescriptorCalculator("ecfp")



X_remedi_full = evaluate_molecular_descriptor_on_dataset(remedi_model, ds)

X_ecfp_full = desc_ecfp.calculate_descriptors(dataset)


for i, task_confs in enumerate(dataset.config.tasks.system_cols):
    print(task_confs.name)
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





cv_params_rf = CVParams(scoring="neg_mean_absolute_error", n_jobs = 1)
remedi_rf = RandomForestLearner(RFParams(n_jobs = -1))
cvr_remedi_rf = CrossValidationRunner(cv_params = cv_params_rf, learner = remedi_rf)

res_remedi_rf = cvr_remedi_rf.run_kfold_repeated_cross_validation(X_remedi, y)
print(res_remedi_rf)
cvr_remedi_rf.write_output(output_directory=eval_dir)


cv_params_rf = CVParams(scoring="neg_mean_absolute_error", n_jobs = 1)
ecfp_rf = RandomForestLearner(RFParams(n_jobs = -1))
cvr_ecfp_rf = CrossValidationRunner(cv_params = cv_params_rf, learner = ecfp_rf)

res_ecfp_rf = cvr_ecfp_rf.run_kfold_repeated_cross_validation(X_ecfp, y)
print(res_ecfp_rf)
cvr_ecfp_rf.write_output(output_directory=eval_dir)
