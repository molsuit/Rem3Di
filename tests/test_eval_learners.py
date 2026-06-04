"""LinearLearner: config round-trip + the three task-type methods.

The shape these tests pin is the contract the runner depends on: each
``fit_predict_*`` returns the right shape, ignoring the validation fold when
the learner has no early-stopping notion (Ridge / LogisticRegression).
Training-only invariant is structural — only X_train + y_train flow into
sklearn ``Pipeline.fit``.
"""

from __future__ import annotations

import numpy as np
import pydantic
import pydantic_yaml as pyd_yaml
import pytest

from threedscriptors.evaluation.benchmark.learners import (
    LearnerConfig,
    LightGBMLearner,
    LightGBMLearnerConfig,
    LinearLearner,
    LinearLearnerConfig,
    MlpLearner,
    MlpLearnerConfig,
    ScalerKind,
)


def _synthetic_regression(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = X @ w + 0.1 * rng.standard_normal(n)
    return X, y


def _synthetic_binary(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_config_yaml_roundtrip(tmp_path) -> None:
    cfg = LinearLearnerConfig(
        ridge_alpha=2.0, logreg_C=0.5, scaler=ScalerKind.standard
    )
    path = tmp_path / "learner.yaml"
    pyd_yaml.to_yaml_file(path, cfg)
    loaded = pyd_yaml.parse_yaml_file_as(LinearLearnerConfig, path)
    assert loaded == cfg
    assert loaded.learner_kind == "linear"


def test_build_returns_linear_learner() -> None:
    cfg = LinearLearnerConfig(ridge_alpha=3.0)
    learner = cfg.build()
    assert isinstance(learner, LinearLearner)
    assert learner.ridge_alpha == 3.0


def test_regression_recovers_signal() -> None:
    X, y = _synthetic_regression(60, 4)
    X_train, X_val, X_test = X[:30], X[30:45], X[45:]
    y_train, y_val, y_test = y[:30], y[30:45], y[45:]
    learner = LinearLearner(ridge_alpha=0.1)
    pred = learner.fit_predict_regression(X_train, y_train, X_val, y_val, X_test)
    assert pred.shape == (len(X_test),)
    # Synthetic linear signal is recoverable above r2=0.5 with Ridge + n=30.
    from sklearn.metrics import r2_score

    assert r2_score(y_test, pred) > 0.5


def test_binary_predicts_class_one_probabilities_in_unit_interval() -> None:
    X, y = _synthetic_binary(60, 4)
    X_train, X_val, X_test = X[:30], X[30:45], X[45:]
    y_train, y_val, y_test = y[:30], y[30:45], y[45:]
    learner = LinearLearner(logreg_C=1.0)
    pred = learner.fit_predict_binary(X_train, y_train, X_val, y_val, X_test)
    assert pred.shape == (len(X_test),)
    assert ((pred >= 0.0) & (pred <= 1.0)).all()
    # X[:,0] > 0 is easy: most predictions should match the rule.
    accuracy = ((pred > 0.5).astype(int) == y_test).mean()
    assert accuracy > 0.7


def test_binary_degenerate_single_class_returns_constant() -> None:
    X = np.random.default_rng(0).standard_normal((20, 3))
    y_all_zero = np.zeros(20)
    learner = LinearLearner()
    pred = learner.fit_predict_binary(
        X, y_all_zero, X, y_all_zero, X[:5]
    )
    assert pred.shape == (5,)
    assert np.allclose(pred, 0.0)


def test_lightgbm_config_roundtrip_and_build(tmp_path) -> None:
    cfg = LightGBMLearnerConfig(n_estimators=300, early_stopping_rounds=10)
    import pydantic_yaml as pyd_yaml

    path = tmp_path / "lgbm.yaml"
    pyd_yaml.to_yaml_file(path, cfg)
    loaded = pyd_yaml.parse_yaml_file_as(LightGBMLearnerConfig, path)
    assert loaded == cfg
    assert isinstance(cfg.build(), LightGBMLearner)


def test_lightgbm_regression_smoke() -> None:
    X, y = _synthetic_regression(120, 4)
    Xtr, Xv, Xte = X[:80], X[80:100], X[100:]
    ytr, yv, yte = y[:80], y[80:100], y[100:]
    learner = LightGBMLearner(n_estimators=200, early_stopping_rounds=10)
    pred = learner.fit_predict_regression(Xtr, ytr, Xv, yv, Xte)
    assert pred.shape == (len(Xte),)
    from sklearn.metrics import r2_score

    assert r2_score(yte, pred) > 0.3  # lightgbm on 80 noisy rows; loose floor


def test_lightgbm_binary_smoke() -> None:
    X, y = _synthetic_binary(120, 4)
    Xtr, Xv, Xte = X[:80], X[80:100], X[100:]
    ytr, yv, yte = y[:80], y[80:100], y[100:]
    learner = LightGBMLearner(n_estimators=200, early_stopping_rounds=10)
    pred = learner.fit_predict_binary(Xtr, ytr, Xv, yv, Xte)
    assert pred.shape == (len(Xte),)
    assert ((pred >= 0.0) & (pred <= 1.0)).all()
    # Probabilities at iter-1 (val AUC saturates immediately) cluster near 0.5,
    # but ranking still recovers the signal — assert AUROC, not accuracy.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(yte, pred) > 0.8


def test_mlp_config_roundtrip_defaults() -> None:
    cfg = MlpLearnerConfig()
    assert cfg.hidden_dims == [256, 128]
    assert cfg.device == "cuda"
    assert cfg.allow_cpu_mlp is False
    # The build() with default CUDA may raise on CPU-only hosts — only the
    # config shape matters here.


def test_mlp_cpu_smoke_regression() -> None:
    # Allow_cpu_mlp=True is required to opt into CPU; max_epochs/patience are
    # tiny to keep this fast on CI without GPUs.
    X, y = _synthetic_regression(100, 4)
    Xtr, Xv, Xte = X[:60], X[60:80], X[80:]
    ytr, yv = y[:60], y[60:80]
    cfg = MlpLearnerConfig(
        hidden_dims=[16, 8],
        max_epochs=20,
        patience=5,
        batch_size=32,
        device="cpu",
        allow_cpu_mlp=True,
    )
    pred = cfg.build().fit_predict_regression(Xtr, ytr, Xv, yv, Xte)
    assert pred.shape == (len(Xte),)
    # The MLP should fit *something* on a clean linear signal even with tiny
    # capacity; we just check the output isn't constant / NaN.
    assert np.isfinite(pred).all()
    assert pred.std() > 0


def test_mlp_cpu_gate_raises_without_allow_flag() -> None:
    with pytest.raises(RuntimeError, match="allow_cpu_mlp"):
        MlpLearner(device="cpu", allow_cpu_mlp=False).fit_predict_regression(
            np.zeros((4, 2), dtype=float),
            np.zeros(4),
            np.zeros((2, 2), dtype=float),
            np.zeros(2),
            np.zeros((1, 2), dtype=float),
        )


def test_discriminated_union_picks_concrete_kind() -> None:
    adapter = pydantic.TypeAdapter(LearnerConfig)
    linear = adapter.validate_python({"learner_kind": "linear"})
    lgbm = adapter.validate_python({"learner_kind": "lightgbm"})
    mlp = adapter.validate_python({"learner_kind": "mlp"})
    assert isinstance(linear, LinearLearnerConfig)
    assert isinstance(lgbm, LightGBMLearnerConfig)
    assert isinstance(mlp, MlpLearnerConfig)


def test_multilabel_runs_per_column() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    Y = np.stack([(X[:, 0] > 0).astype(int), (X[:, 1] > 0).astype(int)], axis=1)
    X_train, X_val, X_test = X[:30], X[30:45], X[45:]
    Y_train, Y_val, Y_test = Y[:30], Y[30:45], Y[45:]
    learner = LinearLearner()
    pred = learner.fit_predict_multilabel(
        X_train, Y_train, X_val, Y_val, X_test
    )
    assert pred.shape == (len(X_test), 2)
    for j in range(2):
        accuracy = ((pred[:, j] > 0.5).astype(int) == Y_test[:, j]).mean()
        assert accuracy > 0.7


def test_null_learner_predicts_train_constant_and_parses():
    from threedscriptors.evaluation.benchmark.learners import (
        NullLearner,
        NullLearnerConfig,
    )

    learner = NullLearner()
    # regression: constant train mean (NaN labels ignored)
    y_train = np.array([1.0, 3.0, np.nan, 5.0])
    pred = learner.fit_predict_regression(
        np.zeros((4, 3)), y_train, np.zeros((1, 3)), np.array([0.0]), np.zeros((2, 3))
    )
    assert pred.shape == (2,)
    assert np.allclose(pred, 3.0)  # mean of [1,3,5]

    # binary: base rate
    pb = learner.fit_predict_binary(
        np.zeros((4, 3)), np.array([0, 1, 1, 1.0]), np.zeros((1, 3)),
        np.array([0.0]), np.zeros((5, 3))
    )
    assert pb.shape == (5,) and np.allclose(pb, 0.75)

    # multilabel: per-column base rate, NaN-aware
    Y = np.array([[1.0, np.nan], [0.0, 1.0], [1.0, 1.0]])
    pm = learner.fit_predict_multilabel(
        np.zeros((3, 3)), Y, np.zeros((1, 3)), np.zeros((1, 2)), np.zeros((4, 3))
    )
    assert pm.shape == (4, 2)
    assert np.allclose(pm[:, 0], 2 / 3) and np.allclose(pm[:, 1], 1.0)

    # discriminated-union round-trip on the string value (not bare ``null``)
    cfg = pyd_yaml.parse_yaml_file_as  # noqa: F841  (sanity: import path stable)
    parsed = pydantic.TypeAdapter(LearnerConfig).validate_python(
        {"learner_kind": "null_baseline"}
    )
    assert isinstance(parsed, NullLearnerConfig)
