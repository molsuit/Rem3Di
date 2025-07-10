
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.data_handling.dataset_analysis import DatasetPostLoadAnalysis
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.evaluation.clustering import UMAPCalculator
from threedscriptors.evaluation.evaluation_utils import evaluate_molecular_descriptor_on_dataset


train_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_train"

val_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_valid"

train_dataset = reload_dataset_pipeline(train_directory).build()
valid_dataset = reload_dataset_pipeline(val_directory).build()

model_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/116-2025_07_09_14_30_02-AAdmet_0depth_0encoder_layers_log"
model = ModelBuilder.from_directory(model_dir).build_model()

train_descriptors = evaluate_molecular_descriptor_on_dataset(model, train_dataset)

val_descriptors = evaluate_molecular_descriptor_on_dataset(model, valid_dataset)


train_pc, umap_calc = UMAPCalculator().get_dimensionality_reduction(train_descriptors, return_fit = True)


valid_pc = umap_calc.transform(val_descriptors)


import matplotlib.pyplot as plt

val_descriptors = val_descriptors.detach().cpu().numpy()
train_descriptors = train_descriptors.detach().cpu().numpy()


fig = plt.figure()

plt.scatter(train_pc[:,0], train_pc[:,1], label = "train")
plt.scatter(valid_pc[:,0], valid_pc[:,1], label = "valid")
plt.legend()
plt.savefig("pc_train_val_admet.png")

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler           # optional but often helpful
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ------------------------------------------------------------------
# 1.  Assemble the inputs
# ------------------------------------------------------------------
X_train = train_descriptors                # shape (n_train , dim)
X_val   = val_descriptors                  # shape (n_val   , dim)

X = np.vstack([X_train, X_val])            # (n_train + n_val, dim)
y = np.concatenate([
        np.zeros(len(X_train), dtype=int), # label 0 → “comes from TRAIN set”
        np.ones (len(X_val  ), dtype=int)  # label 1 → “comes from VAL   set”
])

# ------------------------------------------------------------------
# 2.  Build the k-NN pipeline
# ------------------------------------------------------------------
# • StandardScaler(with_mean=False) is memory-safe for sparse inputs
# • cosine distance is usually sensible for molecular fingerprints
knn_clf = make_pipeline(
    StandardScaler(with_mean=False),
    KNeighborsClassifier(n_neighbors=5, metric='cosine')
)

# ------------------------------------------------------------------
# 3.  Evaluate with stratified 5-fold CV
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc = cross_val_score(
    knn_clf, X, y,
    cv=cv,
    scoring='roc_auc'
)

print(f"ROC-AUC (5-fold): {roc_auc.mean():.3f} ± {roc_auc.std():.3f}")




import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Optional: pip install hyppo
from hyppo.ksample import MMD          # kernel Maximum Mean Discrepancy

# ------------------------------------------------------------
# Helper: distance–ratio statistic
# ------------------------------------------------------------
def distance_ratio(train, val, k=5, metric="euclidean", standardise=True):
    """
    For every validation point:
        ratio = mean_distance_to_k_val_neighbors / mean_distance_to_k_train_neighbors
    Returns the vector of ratios and a summary dict.
    """
    if standardise:
        scaler = StandardScaler(with_mean=True)
        train = scaler.fit_transform(train)
        val   = scaler.transform(val)

    # fit neighbour indices
    nbrs_train = NearestNeighbors(n_neighbors=k, metric=metric).fit(train)
    nbrs_val   = NearestNeighbors(n_neighbors=k, metric=metric).fit(val)

    d_val2val   = nbrs_val.kneighbors(val,   return_distance=True)[0].mean(1)
    d_val2train = nbrs_train.kneighbors(val, return_distance=True)[0].mean(1)

    ratio = d_val2val / (d_val2train + 1e-12)  # avoid /0
    summary = {
        "mean":  ratio.mean(),
        "std":   ratio.std(),
        "median":np.median(ratio),
        "≥1.0":  (ratio >= 1.0).mean(),        # fraction where val–val ≥ val–train
    }
    return ratio, summary

ratios, ratio_stats = distance_ratio(train_descriptors, val_descriptors, k=5)
print("Distance-ratio summary:", ratio_stats)

# ------------------------------------------------------------
# Diagnostic 2: linear (logistic-regression) train/val classifier
# ------------------------------------------------------------
log_reg = LogisticRegressionCV(
    Cs=10,
    cv=5,
    scoring="roc_auc",
    max_iter=5000,
    class_weight="balanced",
    n_jobs=-1,
)
auc = cross_val_score(
    log_reg,
    X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
)
print(f"LogReg ROC-AUC (5-fold): {auc.mean():.3f} ± {auc.std():.3f}")

# ------------------------------------------------------------
# Diagnostic 3a: Maximum-Mean-Discrepancy (MMD) two-sample test
# ------------------------------------------------------------
stat, pvalue = MMD(compute_kernel="rbf", bias=False).test(
    val_descriptors, train_descriptors, reps=1000, workers=-1
)
print(f"MMD statistic = {stat:.4f},   p-value = {pvalue:.4g}")

# ------------------------------------------------------------
# Diagnostic 3b: Energy distance (alternative to MMD, no kernel choice)
# ------------------------------------------------------------
import numpy as np
from scipy.spatial.distance import cdist, pdist

def energy_distance_mv(x: np.ndarray, y: np.ndarray) -> float:
    """
    Multivariate energy distance between two samples.

    Parameters
    ----------
    x, y : array-like, shape (n_samples, n_features)

    Returns
    -------
    float
        Energy distance.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    n, m = len(x), len(y)

    # pairwise Euclidean distances
    d_xy = cdist(x, y, metric="euclidean")          # (n, m)
    d_xx = pdist(x,  metric="euclidean")            # (n·(n-1)/2,)
    d_yy = pdist(y,  metric="euclidean")            # (m·(m-1)/2,)

    term_xy = (2.0 / (n * m)) * d_xy.sum()
    term_xx = (1.0 / (n * (n - 1))) * d_xx.sum()    # multiply by 2 later?
    term_yy = (1.0 / (m * (m - 1))) * d_yy.sum()

    return term_xy - term_xx - term_yy


en_dist = energy_distance_mv(train_descriptors, val_descriptors)

train_en_dist = energy_distance_mv(train_descriptors, train_descriptors)
val_en_dist =  energy_distance_mv(val_descriptors, val_descriptors)

print(f"Multivariate energy distance = {en_dist:.4f}")

print(f"Multivariate energy distance train-train = {train_en_dist:.4f}")
print(f"Multivariate energy distance val-val = {val_en_dist:.4f}")
