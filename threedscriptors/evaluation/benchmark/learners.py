"""Descriptor-probe learners.

A *learner* is one ML model family with one settled config — no HPO. The same
learner handles all three task types via three ``fit_predict_*`` methods; the
runner calls the matching one per benchmark. Each method receives an explicit
``(X_train, y_train, X_val, y_val, X_test)`` so that learners with an early-
stopping notion (LightGBM/MLP) can use ``X_val`` as the stop signal *without*
ever fitting parameters on it — the user-requested invariant is that training
only happens on the train split.

Learner configs are a pydantic ``Annotated`` discriminated union on
``learner_kind`` so the eval yaml lists learners as one block per kind:

    learners:
      - learner_kind: linear
      - learner_kind: lightgbm
        early_stopping_rounds: 50
      - learner_kind: mlp
        hidden_dims: [256, 128]

Ships ``LinearLearner`` (ignores val), ``LightGBMLearner`` (val drives early
stopping via ``eval_set``), and ``MlpLearner`` (val drives early stopping on
held-out loss). All three implement the same three task-type methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
import torch
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


class ScalerKind(StrEnum):
    none = "none"
    standard = "standard"


class Learner(ABC):
    """One model family, three task-type methods.

    The runner picks the task type from the benchmark and calls the matching
    method. Learners that ignore the validation fold (e.g. plain Ridge) still
    accept the kwargs so callers don't branch on learner type.
    """

    @abstractmethod
    def fit_predict_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Return shape (n_test,)."""

    @abstractmethod
    def fit_predict_binary(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Return shape (n_test,) of class-1 probabilities."""

    @abstractmethod
    def fit_predict_multilabel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Return shape (n_test, n_labels) of per-column class-1 probabilities."""


# -- LinearLearner ----------------------------------------------------------
# Ridge / LogReg with StandardScaler. Ignores X_val / y_val: no early stopping
# notion. Train-only fitting is automatic — sklearn's Pipeline.fit() only sees
# X_train + y_train.


def _scaler_for(kind: ScalerKind):
    if kind is ScalerKind.standard:
        return StandardScaler()
    return None


class LinearLearner(Learner):
    def __init__(
        self,
        *,
        ridge_alpha: float = 1.0,
        logreg_C: float = 1.0,
        logreg_max_iter: int = 2000,
        scaler: ScalerKind = ScalerKind.standard,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.logreg_C = logreg_C
        self.logreg_max_iter = logreg_max_iter
        self.scaler = scaler

    def _ridge_pipeline(self):
        s = _scaler_for(self.scaler)
        return (
            make_pipeline(s, Ridge(alpha=self.ridge_alpha))
            if s is not None
            else make_pipeline(Ridge(alpha=self.ridge_alpha))
        )

    def _logreg_pipeline(self, seed: int):
        s = _scaler_for(self.scaler)
        clf = LogisticRegression(
            C=self.logreg_C,
            max_iter=self.logreg_max_iter,
            random_state=seed,
            solver="liblinear",
        )
        return make_pipeline(s, clf) if s is not None else make_pipeline(clf)

    def fit_predict_regression(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        del X_val, y_val  # no early stopping notion
        model = self._ridge_pipeline()
        model.fit(X_train, np.asarray(y_train, dtype=float).reshape(-1))
        return np.asarray(model.predict(X_test), dtype=float).reshape(-1)

    def fit_predict_binary(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        del X_val, y_val
        y = np.asarray(y_train, dtype=float).reshape(-1)
        mask = ~np.isnan(y)
        X = X_train[mask]
        y = y[mask].astype(int)
        if len(np.unique(y)) < 2:
            # degenerate train fold: return the constant majority probability
            const = float(y.mean()) if len(y) > 0 else float("nan")
            return np.full(len(X_test), const)
        model = self._logreg_pipeline(seed)
        model.fit(X, y)
        return np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    def fit_predict_multilabel(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        Y_train = np.asarray(y_train, dtype=float)
        Y_val = np.asarray(y_val, dtype=float)
        n_labels = Y_train.shape[1]
        preds = np.full((len(X_test), n_labels), np.nan, dtype=float)
        for j in range(n_labels):
            preds[:, j] = self.fit_predict_binary(
                X_train, Y_train[:, j], X_val, Y_val[:, j], X_test, seed
            )
        return preds


# -- Pydantic configs -------------------------------------------------------


# -- LightGBMLearner --------------------------------------------------------
# Uses (X_val, y_val) as `eval_set` for early stopping. We do NOT refit on
# train+val with the best_iteration: per the user policy, train-only fitting.
# The early-stopped model is the final model.


class LightGBMLearner(Learner):
    def __init__(
        self,
        *,
        n_estimators: int = 2000,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        self.early_stopping_rounds = early_stopping_rounds

    def _callbacks(self, allow_early_stop: bool = True):
        cb = [log_evaluation(0)]
        if allow_early_stop:
            cb.insert(0, early_stopping(self.early_stopping_rounds))
        return cb

    def fit_predict_regression(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        model = LGBMRegressor(
            **self.params,
            objective="regression",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(
            X_train,
            np.asarray(y_train, dtype=float).reshape(-1),
            eval_set=[(X_val, np.asarray(y_val, dtype=float).reshape(-1))],
            eval_metric="l1",
            callbacks=self._callbacks(),
        )
        return np.asarray(model.predict(X_test), dtype=float).reshape(-1)

    def fit_predict_binary(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        ytr = np.asarray(y_train, dtype=float).reshape(-1)
        yv = np.asarray(y_val, dtype=float).reshape(-1)
        mtr = ~np.isnan(ytr)
        mv = ~np.isnan(yv)
        Xtr = X_train[mtr]
        ytr_i = ytr[mtr].astype(int)
        if len(np.unique(ytr_i)) < 2:
            const = float(ytr_i.mean()) if len(ytr_i) > 0 else float("nan")
            return np.full(len(X_test), const)
        Xv = X_val[mv]
        yv_i = yv[mv].astype(int)
        model = LGBMClassifier(
            **self.params,
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        # Skip early-stopping when val collapses to one class — LGBM warns and
        # the signal is meaningless anyway.
        allow_early = len(np.unique(yv_i)) >= 2 and len(Xv) > 0
        eval_set = [(Xv, yv_i)] if allow_early else [(Xtr, ytr_i)]
        model.fit(
            Xtr,
            ytr_i,
            eval_set=eval_set,
            eval_metric="auc",
            callbacks=self._callbacks(allow_early_stop=allow_early),
        )
        return np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    def fit_predict_multilabel(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        Y_train = np.asarray(y_train, dtype=float)
        Y_val = np.asarray(y_val, dtype=float)
        n_labels = Y_train.shape[1]
        preds = np.full((len(X_test), n_labels), np.nan, dtype=float)
        for j in range(n_labels):
            preds[:, j] = self.fit_predict_binary(
                X_train, Y_train[:, j], X_val, Y_val[:, j], X_test, seed
            )
        return preds


# -- MlpLearner --------------------------------------------------------------
# Small PyTorch MLP. Uses val loss for early stopping with patience; never
# updates parameters from val gradients. CUDA-by-default; CPU is gated so a
# full-panel run can't silently fall back to slow CPU training.


class _Mlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MlpLearner(Learner):
    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 128,
        device: Literal["cuda", "cpu"] = "cuda",
        allow_cpu_mlp: bool = False,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device_str = device
        self.allow_cpu_mlp = allow_cpu_mlp

    # -- internal helpers --

    def _resolve_device(self) -> torch.device:
        if self.device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "MlpLearner requested CUDA but torch.cuda.is_available() is "
                    "False. Set device='cpu' + allow_cpu_mlp=True for a "
                    "diagnostic CPU run."
                )
            return torch.device("cuda")
        if not self.allow_cpu_mlp:
            raise RuntimeError(
                "MlpLearner CPU is gated. Set allow_cpu_mlp=True to opt in."
            )
        return torch.device("cpu")

    @staticmethod
    def _standardize(
        X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train).astype(np.float32)
        Xv = scaler.transform(X_val).astype(np.float32)
        Xte = scaler.transform(X_test).astype(np.float32)
        return Xtr, Xv, Xte

    def _train(
        self,
        model: _Mlp,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mode: Literal["regression", "binary", "multilabel"],
        device: torch.device,
        seed: int,
        *,
        train_mask: np.ndarray | None = None,
        val_mask: np.ndarray | None = None,
    ) -> _Mlp:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        opt = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = len(X_train)
        bs = min(self.batch_size, n) if n >= 2 else max(1, n)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_loss = float("inf")
        stale = 0

        Xv_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
        yv_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
        val_mask_t = (
            torch.from_numpy(val_mask.astype(np.float32)).to(device)
            if val_mask is not None
            else None
        )

        rng = np.random.default_rng(seed)
        for _ in range(self.max_epochs):
            model.train()
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                xb = torch.from_numpy(X_train[idx].astype(np.float32)).to(device)
                yb = torch.from_numpy(y_train[idx].astype(np.float32)).to(device)
                opt.zero_grad()
                out = model(xb)
                if mode == "regression":
                    loss = nn.functional.mse_loss(out, yb)
                elif mode == "binary":
                    loss = nn.functional.binary_cross_entropy_with_logits(out, yb)
                else:
                    assert train_mask is not None
                    mb = torch.from_numpy(
                        train_mask[idx].astype(np.float32)
                    ).to(device)
                    raw = nn.functional.binary_cross_entropy_with_logits(
                        out, yb, reduction="none"
                    )
                    loss = (raw * mb).sum() / mb.sum().clamp_min(1.0)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                out = model(Xv_t)
                if mode == "regression":
                    val_loss = nn.functional.mse_loss(out, yv_t).item()
                elif mode == "binary":
                    val_loss = nn.functional.binary_cross_entropy_with_logits(
                        out, yv_t
                    ).item()
                else:
                    assert val_mask_t is not None
                    raw = nn.functional.binary_cross_entropy_with_logits(
                        out, yv_t, reduction="none"
                    )
                    val_loss = (
                        (raw * val_mask_t).sum() / val_mask_t.sum().clamp_min(1.0)
                    ).item()

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        return model

    # -- task-type methods --

    def fit_predict_regression(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        device = self._resolve_device()
        Xtr, Xv, Xte = self._standardize(X_train, X_val, X_test)
        ymean = float(np.mean(y_train))
        ystd = float(np.std(y_train) or 1.0)
        ytr = ((y_train - ymean) / ystd).astype(np.float32).reshape(-1, 1)
        yv = ((y_val - ymean) / ystd).astype(np.float32).reshape(-1, 1)
        model = _Mlp(Xtr.shape[1], 1, self.hidden_dims, self.dropout).to(device)
        model = self._train(model, Xtr, ytr, Xv, yv, "regression", device, seed)
        with torch.no_grad():
            xte = torch.from_numpy(Xte).to(device)
            raw = model(xte).cpu().numpy().reshape(-1)
        return raw * ystd + ymean

    def fit_predict_binary(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        ytr = np.asarray(y_train, dtype=float).reshape(-1)
        yv = np.asarray(y_val, dtype=float).reshape(-1)
        mtr = ~np.isnan(ytr)
        mv = ~np.isnan(yv)
        if len(np.unique(ytr[mtr])) < 2:
            const = float(ytr[mtr].mean()) if mtr.any() else float("nan")
            return np.full(len(X_test), const)
        device = self._resolve_device()
        Xtr, Xv, Xte = self._standardize(X_train[mtr], X_val[mv], X_test)
        y_tr = ytr[mtr].astype(np.float32).reshape(-1, 1)
        y_v = yv[mv].astype(np.float32).reshape(-1, 1)
        model = _Mlp(Xtr.shape[1], 1, self.hidden_dims, self.dropout).to(device)
        model = self._train(model, Xtr, y_tr, Xv, y_v, "binary", device, seed)
        with torch.no_grad():
            xte = torch.from_numpy(Xte).to(device)
            logits = model(xte).cpu().numpy().reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))

    def fit_predict_multilabel(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        device = self._resolve_device()
        Xtr, Xv, Xte = self._standardize(X_train, X_val, X_test)
        Y_train = np.asarray(y_train, dtype=float)
        Y_val = np.asarray(y_val, dtype=float)
        ytr = np.nan_to_num(Y_train, nan=0.0).astype(np.float32)
        yv = np.nan_to_num(Y_val, nan=0.0).astype(np.float32)
        mask_tr = (~np.isnan(Y_train)).astype(np.float32)
        mask_v = (~np.isnan(Y_val)).astype(np.float32)
        model = _Mlp(
            Xtr.shape[1], Y_train.shape[1], self.hidden_dims, self.dropout
        ).to(device)
        model = self._train(
            model,
            Xtr,
            ytr,
            Xv,
            yv,
            "multilabel",
            device,
            seed,
            train_mask=mask_tr,
            val_mask=mask_v,
        )
        with torch.no_grad():
            xte = torch.from_numpy(Xte).to(device)
            logits = model(xte).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))


# -- Pydantic configs -------------------------------------------------------


class LinearLearnerConfig(BaseModel):
    learner_kind: Literal["linear"] = "linear"
    ridge_alpha: float = 1.0
    logreg_C: float = 1.0
    logreg_max_iter: int = 2000
    scaler: ScalerKind = ScalerKind.standard

    def build(self) -> LinearLearner:
        return LinearLearner(
            ridge_alpha=self.ridge_alpha,
            logreg_C=self.logreg_C,
            logreg_max_iter=self.logreg_max_iter,
            scaler=self.scaler,
        )


class LightGBMLearnerConfig(BaseModel):
    learner_kind: Literal["lightgbm"] = "lightgbm"
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 50

    def build(self) -> LightGBMLearner:
        return LightGBMLearner(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
        )


class MlpLearnerConfig(BaseModel):
    learner_kind: Literal["mlp"] = "mlp"
    hidden_dims: list[int] = [256, 128]
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 128
    device: Literal["cuda", "cpu"] = "cuda"
    allow_cpu_mlp: bool = False

    def build(self) -> MlpLearner:
        return MlpLearner(
            hidden_dims=tuple(self.hidden_dims),
            dropout=self.dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            device=self.device,
            allow_cpu_mlp=self.allow_cpu_mlp,
        )


LearnerConfig = Annotated[
    LinearLearnerConfig | LightGBMLearnerConfig | MlpLearnerConfig,
    Field(discriminator="learner_kind"),
]
