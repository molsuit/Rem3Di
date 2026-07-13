# Train a downstream model

**What you'll do:** fit a predictor (Ridge, LightGBM, or MLP) on frozen Rem3Di
descriptors to predict your own labels, and score it.

## Prerequisites

- [Installed](installation.md) with `--extra eval`.
- Descriptors `X` `(N, D)` from [Evaluate a model](evaluate-a-model.md), and
  labels `y` `(N,)`.

## Option A — quick in-memory fit (Python)

Any learner is a pydantic config with a `.build()`. Each learner exposes
`fit_predict_regression` / `_binary` / `_multiclass` / `_multilabel`.

```python
import numpy as np

# or LightGBMLearnerConfig, MlpLearnerConfig
from remedi.evaluation.benchmark.learners import LinearLearnerConfig
from remedi.evaluation.benchmark.metrics import metric_for
from remedi.data_handling.benchmarks import EvalMetric

# X: (N, D) descriptors, y: (N,) targets — split into train/val/test
idx = np.random.default_rng(0).permutation(len(X))
tr, va, te = np.split(idx, [int(.6*len(X)), int(.8*len(X))])

learner = LinearLearnerConfig(ridge_alpha=1.0).build()
y_pred = learner.fit_predict_regression(
    X[tr], y[tr], X[va], y[va], X[te], seed=0
)

rmse = metric_for(EvalMetric.rmse)(y[te], y_pred)
print(f"RMSE: {rmse:.4f}")
```

Swap the head by swapping the config:

```python
from remedi.evaluation.benchmark.learners import (
    LightGBMLearnerConfig,
    MlpLearnerConfig,
)
learner = LightGBMLearnerConfig(learning_rate=0.03, num_leaves=31).build()
learner = MlpLearnerConfig(
    hidden_dims=[256, 128], dropout=0.2, device="cuda"
).build()
```

Metrics available via `EvalMetric`: `rmse`, `mae`, `r2`, `spearman` (regression);
`auroc`, `auprc`, `balanced_accuracy`, `macro_f1`, `macro_auroc` (classification).

## Option B — from a labelled dataset (CLI)

If your labels live in the dataset's [`tasks/` group](concepts.md#moleculedataset-zarr),
the benchmark panel computes descriptors, reads labels + splits, fits, and scores
in one shot — no manual descriptor handling. This is the same
[`run_eval.py` flow](evaluate-a-model.md#option-b-benchmark-panel-cli): point
`eval_root` at your labelled zarr(s) and list the learners you want. The runner
infers the task type (regression / binary / multiclass / multilabel) from the
`TaskSet` and picks the right metric.

To attach your own targets, build the dataset with a `TaskSet` describing your
label columns (see `remedi/data_handling/dataset/tasks.py` and
[Prepare a dataset](prepare-a-dataset.md)).

## Outputs

- **Option A:** `y_pred` for your test rows + the metric value you compute.
- **Option B:** `results.csv` under `output_root/` — one row per target × learner.

## Next steps

- Notebook: `examples/02_train_downstream_head.ipynb` (runs on synthetic data, no GPU).
- [Prepare a dataset](prepare-a-dataset.md) to attach labels for Option B.
