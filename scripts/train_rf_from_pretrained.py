# Repeated K-Fold CV + tidy Ridge & KRR evaluation (with raw and clip+log MAE)
import numpy as np
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import (
    GridSearchCV, RepeatedKFold, cross_validate, cross_val_predict
)
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import pairwise_distances

from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.evaluation_utils import evaluate_molecular_descriptor_on_dataset
from threedscriptors.model.model_builder import ModelBuilder

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
dataset_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_full"


model_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/backed_up_models/11-2025_08_16_22_40_24-pharma_longer"


#model_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/backed_up_models/8-2025_08_14_09_45_58-pharma_pretraining"
dataset = reload_dataset_pipeline(dataset_path).build()
model = ModelBuilder.from_directory(model_dir).build_remedi_model()
descriptors = evaluate_molecular_descriptor_on_dataset(model, dataset)

task_id = 0  # KSOL
task_idx = np.argwhere(dataset.regression_masks[:, task_id].bool()).squeeze()

X = to_numpy(descriptors[task_idx, :]).astype(np.float64)
y = to_numpy(dataset.regression_targets[task_idx, task_id]).astype(np.float64).ravel()

# Clean NaNs/Infs
ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
X, y = X[ok], y[ok]

# Null baseline
y_mean = y.mean()
baseline_rmse = np.sqrt(np.mean((y - y_mean) ** 2))
baseline_mae = np.mean(np.abs(y - y_mean))
print(f"Null RMSE: {baseline_rmse:.4f} | Null MAE: {baseline_mae:.4f}")

# ---------------------
# CV scheme (Repeated K-Fold)
# ---------------------
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# ---------------------
# RIDGE (pipeline + grid)
# ---------------------
alphas = np.logspace(-3, 5, 13)  # avoid too-small values for stability
ridge_pipe = make_pipeline(
    StandardScaler(),
    VarianceThreshold(threshold=0.0),
    Ridge(solver="svd", fit_intercept=True),
)
ridge_gs = GridSearchCV(
    ridge_pipe,
    {"ridge__alpha": alphas},
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
ridge_gs.fit(X, y)
print("Chosen alpha (Ridge):", ridge_gs.best_params_["ridge__alpha"])

ridge_scores = cross_validate(
    ridge_gs.best_estimator_, X, y, cv=cv,
    scoring={"rmse": "neg_root_mean_squared_error",
             "mae": "neg_mean_absolute_error",
             "r2": "r2"},
    n_jobs=-1,
)
print(f"CV Ridge RMSE: {-ridge_scores['test_rmse'].mean():.4f} ± {ridge_scores['test_rmse'].std():.4f}")
print(f"CV Ridge MAE : {-ridge_scores['test_mae'].mean():.4f} ± {ridge_scores['test_mae'].std():.4f}")
print(f"CV Ridge R2  :  {ridge_scores['test_r2'].mean():.4f} ± {ridge_scores['test_r2'].std():.4f}")

# ---------------------
# KRR (RBF) — gamma via median heuristic on *scaled* X
# ---------------------
Xs = StandardScaler().fit_transform(X)
rng = np.random.default_rng(0)
m = min(4000, len(Xs))
S = Xs[rng.choice(len(Xs), size=m, replace=False)]
d2 = pairwise_distances(S, metric="sqeuclidean")
med = np.median(d2[d2 > 0])
gamma0 = (1.0 / med) if med > 0 else 1.0

krr_pipe = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf"))
krr_grid = {
    "kernelridge__alpha": np.logspace(-4, 2, 13),
    "kernelridge__gamma": gamma0 * np.logspace(-3, 3, 13),
}
krr_gs = GridSearchCV(
    krr_pipe, krr_grid, cv=cv,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)
krr_gs.fit(X, y)
print("Best KRR params:", krr_gs.best_params_)

krr_scores = cross_validate(
    krr_gs.best_estimator_, X, y, cv=cv,
    scoring={"rmse": "neg_root_mean_squared_error",
             "mae": "neg_mean_absolute_error",
             "r2": "r2"},
    n_jobs=-1,
)
print(f"KRR CV RMSE: {-krr_scores['test_rmse'].mean():.4f} ± {krr_scores['test_rmse'].std():.4f}")
print(f"KRR CV MAE : {-krr_scores['test_mae'].mean():.4f} ± {krr_scores['test_mae'].std():.4f}")
print(f"KRR CV R2  :  {krr_scores['test_r2'].mean():.4f} ± {krr_scores['test_r2'].std():.4f}")

# ---------------------

from sklearn.base import clone
import numpy as np
import warnings

# OOF predictions → MAE in clipped+log space
# ---------------------
def repeated_oof_predict(estimator, X, y, cv):
    y_pred = np.zeros_like(y, dtype=float)
    counts = np.zeros_like(y, dtype=float)
    for tr, te in cv.split(X, y):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        y_pred[te] += est.predict(X[te])
        counts[te] += 1
    return y_pred / np.maximum(counts, 1.0)

# use with your RepeatedKFold `cv`
y_pred_ridge_oof = repeated_oof_predict(ridge_gs.best_estimator_, X, y, cv)
y_pred_krr_oof   = repeated_oof_predict(krr_gs.best_estimator_, X, y, cv)

# clip+log MAE (numpy)
def clip_and_log_transform_np(y):
    y = np.asarray(y, dtype=np.float64).ravel()
    y = np.clip(y, 0.0, None)
    return np.log10(y + 1.0)

mae_ridge_cliplog = np.mean(np.abs(clip_and_log_transform_np(y) - clip_and_log_transform_np(y_pred_ridge_oof)))
mae_krr_cliplog   = np.mean(np.abs(clip_and_log_transform_np(y) - clip_and_log_transform_np(y_pred_krr_oof)))

print(f"Ridge MAE (clip+log OOF): {mae_ridge_cliplog:.4f}")
print(f"KRR   MAE (clip+log OOF): {mae_krr_cliplog:.4f}")

# ---------------------
# Test set evaluation (raw + clip+log MAE)
# ---------------------

breakpoint()
test_dataset_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_potency_test_full"
test_dataset = reload_dataset_pipeline(test_dataset_path).build()
test_desc = evaluate_molecular_descriptor_on_dataset(model, test_dataset)

task_idx_test = np.argwhere(test_dataset.regression_masks[:, task_id].bool()).squeeze()
X_test = to_numpy(test_desc[task_idx_test, :]).astype(np.float64)
y_test = to_numpy(test_dataset.regression_targets[task_idx_test, task_id]).astype(np.float64).ravel()

ok_t = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
X_test, y_test = X_test[ok_t], y_test[ok_t]

y_pred_ridge = ridge_gs.best_estimator_.predict(X_test)
y_pred_krr   = krr_gs.best_estimator_.predict(X_test)

print(f"Test MAE (Ridge): {mean_absolute_error(y_test, y_pred_ridge):.4f}")
print(f"Test MAE (KRR)  : {mean_absolute_error(y_test, y_pred_krr):.4f}")

print(f"Test MAE (clip+log, Ridge): {np.mean(np.abs(clip_and_log_transform_np(y_test) - clip_and_log_transform_np(y_pred_ridge))):.4f}")
print(f"Test MAE (clip+log, KRR)  : {np.mean(np.abs(clip_and_log_transform_np(y_test) - clip_and_log_transform_np(y_pred_krr))):.4f}")
