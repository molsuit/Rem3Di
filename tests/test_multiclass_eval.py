"""Multiclass support across the benchmark framework.

Covers:
  * Each learner's ``fit_predict_multiclass`` returns ``(n_test, n_classes)``
    rows that sum to ~1, and recovers a learnable signal.
  * The focal-loss MLP path runs (CPU-gated opt-in).
  * The three multiclass metrics behave on perfect / chance predictions.
  * ``_task_kind`` classifies a single multiclass column as ``"multiclass"``.
  * ``_evaluate_cell`` produces exactly one scored row for the multiclass kind.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from remedi.data_handling.benchmarks import (
    BenchmarkManifest,
    EvalMetric,
    SplitVariant,
)
from remedi.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from remedi.evaluation.benchmark.learners import (
    LightGBMLearner,
    LinearLearner,
    LinearLearnerConfig,
    MlpLearner,
    NullLearner,
)
from remedi.evaluation.benchmark.metrics import (
    balanced_accuracy,
    macro_auroc_ovr,
    macro_f1,
)
from remedi.evaluation.benchmark.runner import _evaluate_cell, _task_kind

N_CLASSES = 5


def _synthetic_multiclass(n_per_class: int, d: int, seed: int = 0):
    """Gaussian blobs, one per class, separated along distinct axes."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in range(N_CLASSES):
        center = np.zeros(d)
        center[c % d] = 4.0
        Xs.append(rng.standard_normal((n_per_class, d)) + center)
        ys.append(np.full(n_per_class, c))
    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(float)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _split(n: int):
    tr = slice(0, int(0.6 * n))
    va = slice(int(0.6 * n), int(0.8 * n))
    te = slice(int(0.8 * n), n)
    return tr, va, te


def _check_proba_shape(p: np.ndarray, n_test: int) -> None:
    assert p.shape == (n_test, N_CLASSES)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(n_test), atol=1e-5)


def test_linear_multiclass_shape_and_signal() -> None:
    X, y = _synthetic_multiclass(40, 6)
    tr, va, te = _split(len(y))
    p = LinearLearner().fit_predict_multiclass(
        X[tr], y[tr], X[va], y[va], X[te], N_CLASSES
    )
    _check_proba_shape(p, len(y[te]))
    acc = (p.argmax(1) == y[te].astype(int)).mean()
    assert acc > 0.8


def test_lightgbm_multiclass_shape() -> None:
    X, y = _synthetic_multiclass(40, 6)
    tr, va, te = _split(len(y))
    p = LightGBMLearner(n_estimators=50).fit_predict_multiclass(
        X[tr], y[tr], X[va], y[va], X[te], N_CLASSES
    )
    _check_proba_shape(p, len(y[te]))


def test_mlp_focal_multiclass_runs_on_cpu() -> None:
    X, y = _synthetic_multiclass(40, 6)
    tr, va, te = _split(len(y))
    learner = MlpLearner(
        hidden_dims=(32,),
        max_epochs=30,
        device="cpu",
        allow_cpu_mlp=True,
        loss="focal",
        focal_gamma=2.0,
    )
    p = learner.fit_predict_multiclass(X[tr], y[tr], X[va], y[va], X[te], N_CLASSES)
    _check_proba_shape(p, len(y[te]))


def test_null_multiclass_returns_base_rates() -> None:
    X, y = _synthetic_multiclass(20, 4)
    tr, va, te = _split(len(y))
    p = NullLearner().fit_predict_multiclass(
        X[tr], y[tr], X[va], y[va], X[te], N_CLASSES
    )
    _check_proba_shape(p, len(y[te]))
    # Every test row gets the identical train base-rate vector.
    assert np.allclose(p, p[0])


def test_missing_class_in_train_still_maps_to_its_column() -> None:
    # Train has classes {0,1,2,3}; class 4 absent. The proba block must still be
    # 5-wide with an all-zero column 4.
    X, y = _synthetic_multiclass(30, 6)
    keep = y < 4
    Xk, yk = X[keep], y[keep]
    p = LinearLearner().fit_predict_multiclass(
        Xk[:80], yk[:80], Xk[80:100], yk[80:100], Xk[100:], N_CLASSES
    )
    assert p.shape[1] == N_CLASSES
    assert np.allclose(p[:, 4], 0.0)


def test_multiclass_metrics_perfect_and_chance() -> None:
    y_true = np.array([0, 1, 2, 3, 4])
    perfect = np.eye(5)[y_true]
    assert balanced_accuracy(y_true, perfect) == 1.0
    assert macro_f1(y_true, perfect) == 1.0
    assert macro_auroc_ovr(y_true, perfect) == 1.0
    # A single test class -> OvR AUROC undefined -> NaN.
    assert np.isnan(
        macro_auroc_ovr(np.zeros(4, dtype=int), np.eye(5)[np.zeros(4, dtype=int)])
    )


def test_task_kind_multiclass() -> None:
    ts = TaskSet.from_list(
        [
            TaskConfig(
                name="chirality_type",
                task_type=TaskType.multiclass,
                scope=TaskScope.system,
            )
        ]
    )
    ds = SimpleNamespace(config=SimpleNamespace(tasks=ts))
    assert _task_kind(ds) == "multiclass"


def test_evaluate_cell_multiclass_single_row() -> None:
    X, y = _synthetic_multiclass(40, 6)
    n = len(y)
    Y = y.reshape(-1, 1)
    tr_i, va_i, te_i = _split(n)
    tr = np.zeros(n, bool)
    va = np.zeros(n, bool)
    te = np.zeros(n, bool)
    tr[tr_i] = True
    va[va_i] = True
    te[te_i] = True

    manifest = BenchmarkManifest(
        dataset_id="chiral_cat",
        metric=EvalMetric.balanced_accuracy,
        split_variant=SplitVariant.stratified,
        source="local",
    )
    rows = _evaluate_cell(
        LinearLearnerConfig(),
        "multiclass",
        (tr, va, te),
        X,
        Y,
        ["chirality_type"],
        manifest,
        "test_descriptor",
        seed=0,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.target_col is None
    assert r.metric_name == EvalMetric.balanced_accuracy.value
    assert 0.0 <= r.metric_value <= 1.0
    assert r.n_test == int(te.sum())
