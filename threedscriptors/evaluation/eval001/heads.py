from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from threedscriptors.evaluation.eval001.metrics import (
    binary_metric,
    multilabel_macro_auroc,
    regression_metric,
)


@dataclass(frozen=True)
class FitResult:
    predictions: np.ndarray
    best_param: str
    notes: str = ""


def _constant_binary(y_train: np.ndarray, n_test: int) -> FitResult | None:
    y = np.asarray(y_train, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return FitResult(np.full(n_test, np.nan), "constant_nan", "no_train_labels")
    unique = np.unique(y)
    if len(unique) < 2:
        return FitResult(np.full(n_test, float(np.mean(y))), "constant", "single_class_train")
    return None


def fit_predict_regression(
    head: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    *,
    max_epochs: int = 100,
    patience: int = 10,
    fast: bool = False,
    mlp_device: str = "cuda",
    allow_cpu_mlp: bool = False,
) -> FitResult:
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)

    if head == "linear":
        best_alpha = 1.0
        best_score = float("inf")
        alpha_grid = (1.0,) if fast else (0.01, 0.1, 1.0, 10.0, 100.0)
        for alpha in alpha_grid:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            score = regression_metric(y_val, pred, "MAE").value
            if score < best_score:
                best_score = score
                best_alpha = alpha
        final = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        final.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        return FitResult(final.predict(X_test), f"alpha={best_alpha}")

    if head == "lightgbm":
        n_estimators = 200 if fast else 2000
        stop_rounds = 20 if fast else 50
        model = LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(stop_rounds), log_evaluation(0)],
        )
        best_iter = int(model.best_iteration_ or min(n_estimators, 200))
        final = LGBMRegressor(
            objective="regression",
            n_estimators=best_iter,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        final.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        return FitResult(final.predict(X_test), f"best_iteration={best_iter}")

    if head == "mlp":
        return _fit_predict_mlp_regression(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            seed,
            max_epochs=max_epochs,
            patience=patience,
            mlp_device=mlp_device,
            allow_cpu_mlp=allow_cpu_mlp,
        )

    raise ValueError(f"Unknown regression head: {head}")


def fit_predict_binary(
    head: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    *,
    max_epochs: int = 100,
    patience: int = 10,
    fast: bool = False,
    mlp_device: str = "cuda",
    allow_cpu_mlp: bool = False,
) -> FitResult:
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    const = _constant_binary(y_train, len(X_test))
    if const is not None:
        return const

    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)
    Xtr = X_train[train_mask]
    ytr = y_train[train_mask].astype(int)
    Xv = X_val[val_mask]
    yv = y_val[val_mask].astype(int)

    if head == "linear":
        best_c = 1.0
        best_score = -float("inf")
        c_grid = (1.0,) if fast else (0.01, 0.1, 1.0, 10.0, 100.0)
        for c in c_grid:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    penalty="l2",
                    max_iter=500 if fast else 2000,
                    random_state=seed,
                    solver="liblinear",
                ),
            )
            model.fit(Xtr, ytr)
            score = binary_metric(yv, model.predict_proba(Xv)[:, 1], "AUROC").value
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_c = c
        final_mask = ~np.isnan(np.concatenate([y_train, y_val]))
        X_final = np.vstack([X_train, X_val])[final_mask]
        y_final = np.concatenate([y_train, y_val])[final_mask].astype(int)
        final = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=best_c,
                penalty="l2",
                max_iter=500 if fast else 2000,
                random_state=seed,
                solver="liblinear",
            ),
        )
        final.fit(X_final, y_final)
        return FitResult(final.predict_proba(X_test)[:, 1], f"C={best_c}")

    if head == "lightgbm":
        n_estimators = 200 if fast else 2000
        stop_rounds = 20 if fast else 50
        model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        callbacks = [log_evaluation(0)]
        if len(np.unique(yv)) == 2:
            callbacks = [early_stopping(stop_rounds), log_evaluation(0)]
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=callbacks)
        best_iter = int(model.best_iteration_ or min(n_estimators, 200))
        final_mask = ~np.isnan(np.concatenate([y_train, y_val]))
        X_final = np.vstack([X_train, X_val])[final_mask]
        y_final = np.concatenate([y_train, y_val])[final_mask].astype(int)
        final = LGBMClassifier(
            objective="binary",
            n_estimators=best_iter,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        final.fit(X_final, y_final)
        return FitResult(final.predict_proba(X_test)[:, 1], f"best_iteration={best_iter}")

    if head == "mlp":
        return _fit_predict_mlp_binary(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            seed,
            max_epochs=max_epochs,
            patience=patience,
            mlp_device=mlp_device,
            allow_cpu_mlp=allow_cpu_mlp,
        )

    raise ValueError(f"Unknown classification head: {head}")


def fit_predict_multilabel(
    head: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    *,
    max_epochs: int = 100,
    patience: int = 10,
    fast: bool = False,
    mlp_device: str = "cuda",
    allow_cpu_mlp: bool = False,
) -> FitResult:
    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)
    n_labels = y_train.shape[1]

    if head in {"linear", "lightgbm"}:
        preds = np.full((len(X_test), n_labels), np.nan, dtype=float)
        notes: list[str] = []
        params: list[str] = []
        for j in range(n_labels):
            res = fit_predict_binary(
                head,
                X_train,
                y_train[:, j],
                X_val,
                y_val[:, j],
                X_test,
                seed,
                max_epochs=max_epochs,
                patience=patience,
                fast=fast,
                mlp_device=mlp_device,
                allow_cpu_mlp=allow_cpu_mlp,
            )
            preds[:, j] = res.predictions
            if res.notes:
                notes.append(f"{j}:{res.notes}")
            params.append(f"{j}:{res.best_param}")
        return FitResult(preds, "|".join(params), ";".join(notes))

    if head == "mlp":
        return _fit_predict_mlp_multilabel(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            seed,
            max_epochs=max_epochs,
            patience=patience,
            mlp_device=mlp_device,
            allow_cpu_mlp=allow_cpu_mlp,
        )

    raise ValueError(f"Unknown multilabel head: {head}")


def _standardize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xv = scaler.transform(X_val).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)
    return Xtr, Xv, Xte


class _Mlp(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _batch_iter(n: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def validate_mlp_device(mlp_device: str, allow_cpu_mlp: bool = False) -> torch.device:
    """Resolve the MLP device, failing closed on accidental CPU runs."""
    if mlp_device not in {"cuda", "cpu"}:
        raise ValueError(f"Unknown MLP device {mlp_device!r}; expected 'cuda' or 'cpu'.")
    if mlp_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MLP head requested CUDA, but torch.cuda.is_available() is false. "
                "Refusing to fall back to CPU because full-panel CPU MLP runs are "
                "prohibitively slow. Run on a GPU host or pass "
                "--mlp-device cpu --allow-cpu-mlp for an explicitly diagnostic CPU run."
            )
        return torch.device("cuda")
    if not allow_cpu_mlp:
        raise RuntimeError(
            "CPU MLP is blocked by default. Use --mlp-device cpu --allow-cpu-mlp "
            "only for a deliberately small diagnostic/smoke, not a final panel."
        )
    return torch.device("cpu")


def _fit_predict_mlp_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    seed,
    *,
    max_epochs: int,
    patience: int,
    mlp_device: str,
    allow_cpu_mlp: bool,
) -> FitResult:
    device = validate_mlp_device(mlp_device, allow_cpu_mlp)
    Xtr, Xv, Xte = _standardize(X_train, X_val, X_test)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) or 1.0)
    ytr = ((y_train - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    yv = ((y_val - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    model = _train_mlp(
        Xtr,
        ytr,
        Xv,
        yv,
        seed,
        "regression",
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    with torch.no_grad():
        xte = torch.from_numpy(Xte).to(device)
        pred = model(xte).cpu().numpy().reshape(-1) * y_std + y_mean
    return FitResult(pred, f"epochs<= {max_epochs};device={device.type}", "")


def _fit_predict_mlp_binary(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    seed,
    *,
    max_epochs: int,
    patience: int,
    mlp_device: str,
    allow_cpu_mlp: bool,
) -> FitResult:
    const = _constant_binary(y_train, len(X_test))
    if const is not None:
        return const
    device = validate_mlp_device(mlp_device, allow_cpu_mlp)
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)
    Xtr, Xv, Xte = _standardize(X_train[train_mask], X_val[val_mask], X_test)
    ytr = y_train[train_mask].astype(np.float32).reshape(-1, 1)
    yv = y_val[val_mask].astype(np.float32).reshape(-1, 1)
    model = _train_mlp(
        Xtr,
        ytr,
        Xv,
        yv,
        seed,
        "binary",
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    with torch.no_grad():
        xte = torch.from_numpy(Xte).to(device)
        logits = model(xte).cpu().numpy().reshape(-1)
    return FitResult(1.0 / (1.0 + np.exp(-logits)), f"epochs<= {max_epochs};device={device.type}", "")


def _fit_predict_mlp_multilabel(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    seed,
    *,
    max_epochs: int,
    patience: int,
    mlp_device: str,
    allow_cpu_mlp: bool,
) -> FitResult:
    device = validate_mlp_device(mlp_device, allow_cpu_mlp)
    Xtr, Xv, Xte = _standardize(X_train, X_val, X_test)
    ytr = np.nan_to_num(y_train, nan=0.0).astype(np.float32)
    yv = np.nan_to_num(y_val, nan=0.0).astype(np.float32)
    mask_tr = (~np.isnan(y_train)).astype(np.float32)
    mask_v = (~np.isnan(y_val)).astype(np.float32)
    model = _train_mlp(
        Xtr,
        ytr,
        Xv,
        yv,
        seed,
        "multilabel",
        train_mask=mask_tr,
        val_mask=mask_v,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    with torch.no_grad():
        xte = torch.from_numpy(Xte).to(device)
        logits = model(xte).cpu().numpy()
    return FitResult(1.0 / (1.0 + np.exp(-logits)), f"epochs<= {max_epochs};device={device.type}", "")


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    mode: str,
    *,
    train_mask: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> _Mlp:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _Mlp(X_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = min(128, len(X_train)) if len(X_train) >= 256 else max(1, len(X_train))
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

    for epoch in range(max_epochs):
        model.train()
        for batch in _batch_iter(len(X_train), batch_size, seed + epoch):
            xb = torch.from_numpy(X_train[batch].astype(np.float32)).to(device)
            yb = torch.from_numpy(y_train[batch].astype(np.float32)).to(device)
            opt.zero_grad()
            out = model(xb)
            if mode == "regression":
                loss = torch.nn.functional.mse_loss(out, yb)
            elif mode == "binary":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yb)
            else:
                mb = torch.from_numpy(train_mask[batch].astype(np.float32)).to(device)  # type: ignore[index]
                raw = torch.nn.functional.binary_cross_entropy_with_logits(out, yb, reduction="none")
                loss = (raw * mb).sum() / mb.sum().clamp_min(1.0)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(Xv_t)
            if mode == "regression":
                val_loss = torch.nn.functional.mse_loss(out, yv_t).item()
            elif mode == "binary":
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yv_t).item()
            else:
                raw = torch.nn.functional.binary_cross_entropy_with_logits(out, yv_t, reduction="none")
                val_loss = ((raw * val_mask_t).sum() / val_mask_t.sum().clamp_min(1.0)).item()  # type: ignore[union-attr]
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model
