"""Descriptor-probe learners.

A *learner* is one ML model family with one settled config — no HPO. Each is a
pydantic model: its fields are its parameters (declared once, read from the
panel yaml) and its ``fit_predict_*`` methods fit and predict. The runner calls
the method matching the benchmark's task kind with an explicit
``(X_train, y_train, X_val, y_val, X_test)``, so learners with an early-stopping
notion (LightGBM, MLP) can use ``X_val`` as the stop signal *without* ever
fitting parameters on it — training only happens on the train split.

What every learner shares lives in :class:`LearnerBase`: rows whose label is
NaN are dropped from the fit folds (a sparse multi-target benchmark leaves NaN
where a molecule has no measurement; the test fold is predicted in full and
masked when scored), a train fold with fewer than two classes falls back to
its base rate, and a multilabel task is one binary fit per column unless the
learner fits the columns jointly (the MLP, with a masked loss).

The learners form a pydantic ``Annotated`` union on ``kind``. Each
carries a ``name`` (defaulting to its kind) that identifies it in result rows
and artifact paths, so two of one kind can sit side by side in a panel:

    learners:
      - kind: linear
      - kind: lightgbm
        early_stopping_rounds: 50
      - kind: mlp
        name: mlp_focal
        loss: focal
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
import torch
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from remedi_eval.benchmark.losses import FocalLoss


class ScalerKind(StrEnum):
    none = "none"
    standard = "standard"


@dataclass(frozen=True)
class Fold:
    """The labelled rows of one fold: features and 1-d float labels."""

    features: np.ndarray
    labels: np.ndarray

    @classmethod
    def labelled(cls, features: np.ndarray, labels: np.ndarray) -> Fold:
        """The rows of ``features`` whose label is not NaN."""
        labels = np.asarray(labels, dtype=float).reshape(-1)
        keep = ~np.isnan(labels)
        return cls(features=np.asarray(features)[keep], labels=labels[keep])

    def classes(self) -> np.ndarray:
        return np.unique(self.labels.astype(int))


def _base_rates(labels: np.ndarray, n_classes: int) -> np.ndarray:
    rates = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
    return rates / rates.sum() if rates.sum() else rates


def _mean_or_nan(labels: np.ndarray) -> float:
    return float(labels.mean()) if labels.size else float("nan")


def _align_proba(
    proba: np.ndarray, model_classes: np.ndarray, n_classes: int
) -> np.ndarray:
    """Reindex an sklearn ``predict_proba`` block to a fixed ``0..n_classes-1``
    column order. Classes the fitted model never saw get an all-zero column."""
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, c in enumerate(np.asarray(model_classes, dtype=int)):
        if 0 <= c < n_classes:
            out[:, c] = proba[:, j]
    return out


class LearnerBase(BaseModel):
    """One model family, four task-type methods.

    The public ``fit_predict_*`` methods drop NaN-label rows and handle a
    degenerate train fold; the learner implements the ``_regression``,
    ``_binary`` and ``_multiclass`` fits on the cleaned folds. ``name`` labels
    the learner in ``results.csv`` (``learner_name``) and in the per-cell
    artifact directory ``<descriptor>__<name>``; the panel validates that the
    names are unique.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str

    @abstractmethod
    def _regression(
        self, train: Fold, valid: Fold, X_test: np.ndarray, seed: int
    ) -> np.ndarray:
        """Predictions, shape ``(n_test,)``."""

    @abstractmethod
    def _binary(
        self, train: Fold, valid: Fold, X_test: np.ndarray, seed: int
    ) -> np.ndarray:
        """Class-1 probabilities, shape ``(n_test,)``; ``train`` has both classes."""

    @abstractmethod
    def _multiclass(
        self, train: Fold, valid: Fold, X_test: np.ndarray, n_classes: int, seed: int
    ) -> np.ndarray:
        """Probabilities ``(n_test, n_classes)``; ``train`` has two or more classes."""

    def fit_predict_regression(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        """Return shape (n_test,)."""
        return self._regression(
            Fold.labelled(X_train, y_train), Fold.labelled(X_val, y_val), X_test, seed
        )

    def fit_predict_binary(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        """Return shape (n_test,) of class-1 probabilities.

        A train fold with one class (or none) predicts its base rate.
        """
        train = Fold.labelled(X_train, y_train)
        if len(train.classes()) < 2:
            return np.full(len(X_test), _mean_or_nan(train.labels))
        return self._binary(train, Fold.labelled(X_val, y_val), X_test, seed)

    def fit_predict_multilabel(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        """Return shape (n_test, n_labels): one binary fit per label column."""
        Y_train = np.asarray(y_train, dtype=float)
        Y_val = np.asarray(y_val, dtype=float)
        return np.stack(
            [
                self.fit_predict_binary(
                    X_train, Y_train[:, column], X_val, Y_val[:, column], X_test, seed
                )
                for column in range(Y_train.shape[1])
            ],
            axis=1,
        )

    def fit_predict_multiclass(
        self, X_train, y_train, X_val, y_val, X_test, n_classes: int, seed: int = 0
    ) -> np.ndarray:
        """Return shape (n_test, n_classes) of class probabilities.

        ``y_train`` / ``y_val`` are 1-D integer class labels in ``0..n_classes-1``;
        the returned columns are aligned to that fixed class ordering so a class
        absent from a (small) train fold still maps to its own column. A train
        fold with one class (or none) predicts its base rates.
        """
        train = Fold.labelled(X_train, y_train)
        if len(train.classes()) < 2:
            return np.tile(_base_rates(train.labels, n_classes), (len(X_test), 1))
        return self._multiclass(
            train, Fold.labelled(X_val, y_val), X_test, n_classes, seed
        )


# -- LinearLearner ----------------------------------------------------------
# Ridge / logistic regression behind an optional StandardScaler. Ignores the
# validation fold: no early-stopping notion. Train-only fitting is automatic —
# sklearn's Pipeline.fit() only sees the train fold.


class LinearLearner(LearnerBase):
    kind: Literal["linear"] = "linear"
    name: str = "linear"
    ridge_alpha: float = 1.0
    logreg_C: float = 1.0
    logreg_max_iter: int = 2000
    scaler: ScalerKind = ScalerKind.standard

    def _pipeline(self, estimator) -> Pipeline:
        scaler = StandardScaler() if self.scaler is ScalerKind.standard else None
        return make_pipeline(scaler or "passthrough", estimator)

    def _logistic(self, seed: int, solver: Literal["liblinear", "lbfgs"]) -> Pipeline:
        return self._pipeline(
            LogisticRegression(
                C=self.logreg_C,
                max_iter=self.logreg_max_iter,
                random_state=seed,
                solver=solver,
            )
        )

    def _regression(self, train, valid, X_test, seed) -> np.ndarray:
        model = self._pipeline(Ridge(alpha=self.ridge_alpha))
        model.fit(train.features, train.labels)
        return np.asarray(model.predict(X_test), dtype=float).reshape(-1)

    def _binary(self, train, valid, X_test, seed) -> np.ndarray:
        model = self._logistic(seed, "liblinear")
        model.fit(train.features, train.labels.astype(int))
        return np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    def _multiclass(self, train, valid, X_test, n_classes, seed) -> np.ndarray:
        # lbfgs fits a single multinomial (softmax) model for multiclass.
        model = self._logistic(seed, "lbfgs")
        model.fit(train.features, train.labels.astype(int))
        return _align_proba(
            np.asarray(model.predict_proba(X_test), dtype=float),
            model.classes_,
            n_classes,
        )


# -- LightGBMLearner --------------------------------------------------------
# Uses the validation fold as `eval_set` for early stopping. We do NOT refit on
# train+val with the best_iteration: per the user policy, train-only fitting.
# The early-stopped model is the final model.


class LightGBMLearner(LearnerBase):
    kind: Literal["lightgbm"] = "lightgbm"
    name: str = "lightgbm"
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.9
    # LightGBM ignores `subsample` entirely unless `subsample_freq > 0`.
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 50

    def _model[Model: (LGBMRegressor, LGBMClassifier)](
        self, model_class: type[Model], seed: int
    ) -> Model:
        """The estimator with this learner's parameters; a classifier derives
        its objective (binary or multiclass) and class count from the train
        fold's labels."""
        return model_class(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )

    def _callbacks(self, allow_early_stop: bool = True):
        callbacks = [log_evaluation(0)]
        if allow_early_stop:
            # LightGBM also tracks the objective's default metric (binary_logloss,
            # l2) beside ``eval_metric`` and would otherwise stop on whichever of
            # them stalls first; ``eval_metric`` is listed first.
            callbacks.insert(
                0, early_stopping(self.early_stopping_rounds, first_metric_only=True)
            )
        return callbacks

    def _fit_classifier(
        self, train: Fold, valid: Fold, eval_metric: str, seed: int
    ) -> LGBMClassifier:
        """Early-stop on the validation fold only when it carries two or more
        classes; otherwise the signal is meaningless (and LightGBM warns)."""
        model = self._model(LGBMClassifier, seed)
        labels = train.labels.astype(int)
        allow_early = len(valid.classes()) >= 2
        eval_set = (
            [(valid.features, valid.labels.astype(int))]
            if allow_early
            else [(train.features, labels)]
        )
        model.fit(
            train.features,
            labels,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=self._callbacks(allow_early_stop=allow_early),
        )
        return model

    def _regression(self, train, valid, X_test, seed) -> np.ndarray:
        model = self._model(LGBMRegressor, seed)
        model.fit(
            train.features,
            train.labels,
            eval_set=[(valid.features, valid.labels)],
            eval_metric="l1",
            callbacks=self._callbacks(),
        )
        return np.asarray(model.predict(X_test), dtype=float).reshape(-1)

    def _binary(self, train, valid, X_test, seed) -> np.ndarray:
        model = self._fit_classifier(train, valid, "auc", seed)
        return np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    def _multiclass(self, train, valid, X_test, n_classes, seed) -> np.ndarray:
        # A valid row whose class the train fold lacks cannot be scored by the
        # fitted label encoder (LightGBM raises "previously unseen labels"), and
        # carries no early-stopping signal for a class the model cannot predict.
        seen = np.isin(valid.labels, train.classes())
        valid = Fold(features=valid.features[seen], labels=valid.labels[seen])
        # ``logloss`` is binary_logloss for two train classes, multi_logloss
        # for more.
        model = self._fit_classifier(train, valid, "logloss", seed)
        return _align_proba(
            np.asarray(model.predict_proba(X_test), dtype=float),
            model.classes_,
            n_classes,
        )


# -- MlpLearner --------------------------------------------------------------
# Small PyTorch MLP. Uses val loss for early stopping with patience; never
# updates parameters from val gradients. CUDA-by-default; CPU is gated so a
# full-panel run can't silently fall back to slow CPU training.


class _Mlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)]
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MlpMode = Literal["regression", "binary", "multilabel", "multiclass"]


def _standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train).astype(np.float32),
        scaler.transform(X_val).astype(np.float32),
        scaler.transform(X_test).astype(np.float32),
    )


class MlpLearner(LearnerBase):
    kind: Literal["mlp"] = "mlp"
    name: str = "mlp"
    hidden_dims: list[int] = [256, 128]
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 128
    device: Literal["cuda", "cpu"] = "cuda"
    allow_cpu_mlp: bool = False
    # Multiclass objective: "cross_entropy" or "focal" (Lin et al. 2017). Focal
    # down-weights the dominant classes — set it for imbalanced multiclass
    # panels like chiral_cat; ``focal_gamma=0`` is cross-entropy. Ignored by the
    # regression/binary/multilabel paths.
    loss: Literal["cross_entropy", "focal"] = "cross_entropy"
    focal_gamma: float = 2.0

    def _resolve_device(self) -> torch.device:
        if self.device == "cuda":
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

    def _loss(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        mode: MlpMode,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-mode objective. Multiclass picks focal vs cross-entropy by config;
        multilabel applies the per-label validity mask; the rest are plain."""
        if mode == "regression":
            return nn.functional.mse_loss(out, target)
        if mode == "binary":
            return nn.functional.binary_cross_entropy_with_logits(out, target)
        if mode == "multiclass":
            if self.loss == "focal":
                return FocalLoss(gamma=self.focal_gamma)(out, target)
            return nn.functional.cross_entropy(out, target)
        # multilabel: masked BCE over the per-label validity mask.
        assert mask is not None
        raw = nn.functional.binary_cross_entropy_with_logits(
            out, target, reduction="none"
        )
        return (raw * mask).sum() / mask.sum().clamp_min(1.0)

    def _train(
        self,
        output_dim: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mode: MlpMode,
        device: torch.device,
        seed: int,
        *,
        train_mask: np.ndarray | None = None,
        val_mask: np.ndarray | None = None,
    ) -> _Mlp:
        """Seed, build and train a fresh network; return it at its best epoch.

        The network is built *after* seeding so its initial weights (and the
        dropout masks) depend on ``seed`` alone, not on whatever drew from the
        global torch RNG before this fit.
        """
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = _Mlp(X_train.shape[1], output_dim, self.hidden_dims, self.dropout).to(
            device
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = len(X_train)
        batch = min(self.batch_size, n) if n >= 2 else max(1, n)
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        best_loss = float("inf")
        stale = 0

        # Multiclass labels are integer class indices (long); every other mode
        # uses float targets.
        label_dtype = np.int64 if mode == "multiclass" else np.float32
        valid_features = torch.from_numpy(X_val.astype(np.float32)).to(device)
        valid_labels = torch.from_numpy(y_val.astype(label_dtype)).to(device)
        valid_mask = (
            torch.from_numpy(val_mask.astype(np.float32)).to(device)
            if val_mask is not None
            else None
        )

        rng = np.random.default_rng(seed)
        for _ in range(self.max_epochs):
            model.train()
            order = rng.permutation(n)
            for start in range(0, n, batch):
                index = order[start : start + batch]
                features = torch.from_numpy(X_train[index].astype(np.float32)).to(
                    device
                )
                labels = torch.from_numpy(y_train[index].astype(label_dtype)).to(device)
                optimizer.zero_grad()
                mask = (
                    torch.from_numpy(train_mask[index].astype(np.float32)).to(device)
                    if train_mask is not None
                    else None
                )
                loss = self._loss(model(features), labels, mode, mask)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = self._loss(
                    model(valid_features), valid_labels, mode, valid_mask
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

    @staticmethod
    def _predict(model: _Mlp, X_test: np.ndarray, device: torch.device) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    def _regression(self, train, valid, X_test, seed) -> np.ndarray:
        device = self._resolve_device()
        Xtr, Xv, Xte = _standardize(train.features, valid.features, X_test)
        mean = float(np.mean(train.labels))
        scale = float(np.std(train.labels) or 1.0)
        ytr = ((train.labels - mean) / scale).astype(np.float32).reshape(-1, 1)
        yv = ((valid.labels - mean) / scale).astype(np.float32).reshape(-1, 1)
        model = self._train(1, Xtr, ytr, Xv, yv, "regression", device, seed)
        return self._predict(model, Xte, device).reshape(-1) * scale + mean

    def _binary(self, train, valid, X_test, seed) -> np.ndarray:
        device = self._resolve_device()
        Xtr, Xv, Xte = _standardize(train.features, valid.features, X_test)
        model = self._train(
            1,
            Xtr,
            train.labels.astype(np.float32).reshape(-1, 1),
            Xv,
            valid.labels.astype(np.float32).reshape(-1, 1),
            "binary",
            device,
            seed,
        )
        return expit(self._predict(model, Xte, device).reshape(-1))

    def fit_predict_multilabel(
        self, X_train, y_train, X_val, y_val, X_test, seed: int = 0
    ) -> np.ndarray:
        """One network over every label, NaN labels masked out of the loss."""
        device = self._resolve_device()
        Xtr, Xv, Xte = _standardize(X_train, X_val, X_test)
        Y_train = np.asarray(y_train, dtype=float)
        Y_val = np.asarray(y_val, dtype=float)
        model = self._train(
            Y_train.shape[1],
            Xtr,
            np.nan_to_num(Y_train, nan=0.0).astype(np.float32),
            Xv,
            np.nan_to_num(Y_val, nan=0.0).astype(np.float32),
            "multilabel",
            device,
            seed,
            train_mask=(~np.isnan(Y_train)).astype(np.float32),
            val_mask=(~np.isnan(Y_val)).astype(np.float32),
        )
        return expit(self._predict(model, Xte, device))

    def _multiclass(self, train, valid, X_test, n_classes, seed) -> np.ndarray:
        device = self._resolve_device()
        Xtr, Xv, Xte = _standardize(train.features, valid.features, X_test)
        model = self._train(
            n_classes,
            Xtr,
            train.labels.astype(np.int64),
            Xv,
            valid.labels.astype(np.int64),
            "multiclass",
            device,
            seed,
        )
        logits = torch.from_numpy(self._predict(model, Xte, device))
        return np.asarray(torch.softmax(logits, dim=-1).numpy(), dtype=float)


# -- NullLearner ------------------------------------------------------------
# Featureless reference: predict the train-set constant (mean for regression,
# base rate for classification), ignoring X entirely. A constant score gives
# AUROC=0.5 and AUPRC=base-rate, the canonical "no-skill" floor every real
# descriptor must clear. Descriptor-independent, so one run per dataset suffices.


class NullLearner(LearnerBase):
    # "null_baseline" not "null": a bare ``null`` in yaml parses to None and
    # would break discriminated-union resolution.
    kind: Literal["null_baseline"] = "null_baseline"
    name: str = "null_baseline"

    def _regression(self, train, valid, X_test, seed) -> np.ndarray:
        return np.full(len(X_test), _mean_or_nan(train.labels))

    def _binary(self, train, valid, X_test, seed) -> np.ndarray:
        return np.full(len(X_test), _mean_or_nan(train.labels))

    def _multiclass(self, train, valid, X_test, n_classes, seed) -> np.ndarray:
        return np.tile(_base_rates(train.labels, n_classes), (len(X_test), 1))


Learner = Annotated[
    LinearLearner | LightGBMLearner | MlpLearner | NullLearner,
    Field(discriminator="kind"),
]
