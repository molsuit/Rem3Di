# Repeated K-Fold CV + tidy Ridge & KRR evaluation (with raw and clip+log MAE)
import warnings

import numpy as np
import torch
from rdkit import RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.regression.lgbm import (
    LGBMParams,
)
from threedscriptors.evaluation.regression.random_forest import (
    RFParams,
    rf_repeated_kfold_cv,
)
from threedscriptors.evaluation.regression.ridge import ridge_repeated_kfold_cv

from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.evaluation.regression.featurization import calculate_mol_features
from threedscriptors.evaluation.regression.utils import cv_results_to_nested_dict
from threedscriptors.model.model_builder import ModelBuilder

RDLogger.DisableLog("rdApp.warning")

warnings.filterwarnings("ignore")


# ---------------------
# Utils
# ---------------------
def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    try:
        return a.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(a)


def clip_and_log_transform_np(y):
    y = np.asarray(y, dtype=np.float64).ravel()
    y = np.clip(y, 0.0, None)
    return np.log10(y + 1.0)


# ---------------------
# Seeds
# ---------------------
torch.manual_seed(0)
np.random.seed(0)

# ---------------------
# Data (train/val pool)
# ---------------------
dataset_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/pharma_properties_train"


test_dataset_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/pharma_properties_test"

model_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/backed_up_models/11-2025_08_16_22_40_24-pharma_longer"

dataset = reload_dataset_pipeline(dataset_path).build()
test_dataset = reload_dataset_pipeline(test_dataset_path).build()

model = ModelBuilder.from_directory(model_dir).build_remedi_model()
descriptors = evaluate_molecular_descriptor_on_dataset(model, dataset)

test_descriptors = evaluate_molecular_descriptor_on_dataset(model, test_dataset)

test_performances = {}

val_performances = {}


def get_task_input_and_labels(dataset, descriptors, task_idx):

    task_sample_idx = np.argwhere(
        dataset.regression_masks[:, task_idx].bool()
    ).squeeze()

    X_remedi = to_numpy(descriptors[task_sample_idx, :]).astype(np.float64)
    y = (
        to_numpy(dataset.regression_targets[task_sample_idx, task_idx])
        .astype(np.float64)
        .ravel()
    )

    smiles = [dataset.smiles_list[i] for i in task_sample_idx]

    X_ci_descriptor = calculate_mol_features(smiles, "ecfp")

    return X_remedi, X_ci_descriptor, y


def eval_one(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    return {name: {"MAE": mae, "MSE": mse, "R2": r2}}


for task_idx, task in enumerate(dataset.dataset_config.tasks):

    print(f"{10*"="} \n Starting benchmark for task {task.task_name}: \n {10*"="}")

    X_remedi, X_ci_descriptor, y = get_task_input_and_labels(
        dataset, descriptors, task_idx
    )

    assert X_remedi.shape[0] == X_ci_descriptor.shape[0] == y.shape[0]

    res_remedi_unscaled = ridge_repeated_kfold_cv(
        X_remedi,
        y,
        n_splits=5,
        n_repeats=5,
        inner_splits=5,
        alphas=np.logspace(-6, 6, 25),  # tweak as needed
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    remedi_scaler = StandardScaler(with_mean=True, with_std=True)

    res_remedi_scaled = ridge_repeated_kfold_cv(
        X_remedi,
        y,
        n_splits=5,
        n_repeats=5,
        inner_splits=5,
        alphas=np.logspace(-6, 6, 25),  # tweak as needed
        scoring="neg_mean_absolute_error",
        scaler=remedi_scaler,
        n_jobs=-1,
    )

    res_ci = ridge_repeated_kfold_cv(
        X_ci_descriptor,
        y,
        n_splits=5,
        n_repeats=5,
        inner_splits=5,
        alphas=np.logspace(-6, 6, 25),  # tweak as needed
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    remedi_lgbm_params = LGBMParams(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=5.0,
        # add subsample_freq=1 to your dataclass if it isn’t there yet
    )

    # res_lgbm_remedi = lightgbm_repeated_kfold_cv(X_remedi, y,
    #                                n_splits=5, n_repeats=3,
    #                                lgbm_params=remedi_lgbm_params,
    #                                tune=False, use_early_stopping=True,early_stopping_rounds=200)
    #
    #   #res_lgbm_ci = lightgbm_repeated_kfold_cv(X_ci_descriptor, y,
    #                                n_splits=5, n_repeats=3,
    #                                lgbm_params=LGBMParams(),
    #                                tune=False,use_early_stopping=True,early_stopping_rounds=200)
    #

    rf_params_remedi = RFParams(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
    )

    res_rf_remedi = rf_repeated_kfold_cv(
        X_remedi,
        y,
        n_splits=5,
        n_repeats=5,
        rf_params=rf_params_remedi,
        tune=False,
    )

    rf_bits_params = RFParams(
        n_estimators=800,  # 600–1200 is a good band
        max_depth=12,  # cap depth to curb variance (or None with higher min_leaf)
        max_features="sqrt",  # ~sqrt(2048)≈45 features per split; good randomness
        min_samples_leaf=8,  # avoid tiny leaves on sparse bits
        min_samples_split=10,  # slightly stricter split condition
        bootstrap=True,
        max_samples=0.8,  # subsample rows for speed + regularization
        oob_score=False,  # set True if you want a quick OOB sanity check; off if using CV
        n_jobs=-1,
    )

    res_rf_ci = rf_repeated_kfold_cv(
        X_ci_descriptor,
        y,
        n_splits=5,
        n_repeats=5,
        rf_params=rf_bits_params,
        tune=False,
    )

    cv_map = {
        "Ridge REM3DI (unscaled)": res_remedi_unscaled,
        "Ridge REM3DI (scaled)": res_remedi_scaled,
        "RF REM3DI": res_rf_remedi,
        "Ridge ECFP": res_ci,
        "RF ECFP": res_rf_ci,
    }

    cv_dict_simple = cv_results_to_nested_dict(
        task_name=task.task_name,
        model_to_cvresult=cv_map,
        round_to=6,  # optional rounding
        include_std=False,
        include_folds=False,
    )

    print(cv_dict_simple)

    val_performances.update(cv_dict_simple)

    # The refit model on all data:
    remedi_ridge_unscaled_model = res_remedi_unscaled.final_model
    remedi_ridge_scaled_model = res_remedi_scaled.final_model
    remedi_rf_model = res_rf_remedi.final_model
    ecfp_ridge_model = res_ci.final_model
    ecfp_rf_model = res_rf_ci.final_model

    X_remedi_test, X_ci_test, y_test = get_task_input_and_labels(
        test_dataset, test_descriptors, task_idx
    )

    # Null baseline on test labels
    y_mean_test = float(np.mean(y_test))
    null_mse = float(np.mean((y_test - y_mean_test) ** 2))
    null_mae = float(np.mean(np.abs(y_test - y_mean_test)))

    task_test_performance = {
        "Null (test mean)": {"MAE": null_mae, "MSE": null_mse, "R2": 0.0}
    }

    # REM3DI descriptors
    task_test_performance.update(
        eval_one(
            "Ridge REM3DI (unscaled)",
            remedi_ridge_unscaled_model,
            X_remedi_test,
            y_test,
        )
    )
    task_test_performance.update(
        eval_one(
            "Ridge REM3DI (scaled)", remedi_ridge_scaled_model, X_remedi_test, y_test
        )
    )
    task_test_performance.update(
        eval_one("RF REM3DI", remedi_rf_model, X_remedi_test, y_test)
    )

    # ECFP (cheminformatics) descriptors
    task_test_performance.update(
        eval_one("Ridge ECFP", ecfp_ridge_model, X_ci_test, y_test)
    )
    task_test_performance.update(eval_one("RF ECFP", ecfp_rf_model, X_ci_test, y_test))

    print(task_test_performance)

    test_performances.update({task.task_name:task_test_performance})


import yaml


def dump_yaml(obj, path: str, sort_keys: bool = False):
    """Write dict to YAML with safe_dump."""
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=sort_keys)


# --- usage inside your loop (per task) ---
# results_rows = [...]  # as you already build it

dump_yaml(test_performances, "test_metrics.yaml")
dump_yaml(val_performances, "val_metrics.yaml")
