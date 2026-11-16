"""Train a REM3DI model from scratch on the chiral-type classification task.

Supervised counterpart of ``run_online_embedding_denoising_pretraining.py``. The
frozen MACE foundation model featurizes raw atoms on the fly
(``PreprocessorWithAtomicEmbedding``); the trainable pseudoscalar preprocessor +
transformer encoder + classification head are optimized end to end with a
focal-loss objective (the right choice for the heavy achiral/central vs.
axial/helical/planar imbalance).

The architecture is the *unified* ``kind: regression`` config whose single head
carries ``n_classes`` — that and ``mace_config`` (which switches the preprocessor
to on-the-fly MACE) are the only things that make it a classifier. The
train/valid split is the dataset's ``default_split`` column of ``table.parquet``
(the stored stratified split; ``split_config`` in the training yaml is not used
for this task), and the labels are its multiclass label column, both read
through :func:`remedi_eval.protocol.load_benchmark`; the test fold is held out
for the eval framework's ``benchmark_panel``, whose multiclass cells carry the
per-class report.

Best-by-balanced-accuracy checkpoints are written in the checkpoint layout
(:mod:`remedi.model.checkpoint`: encoder, preprocessors and the classification
head) so the trained model drops straight into the eval framework as a
``remedi`` descriptor.

Run with:
    uv run python scripts/train_chiral_classifier.py \\
        --training_config configs/training/chiral_cat/training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset

from remedi.configuration.architecture_config import (
    ArchitectureConfig,
    RegressionArchitectureConfig,
)
from remedi.configuration.training_config import ClassifierTrainingConfig
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, save_checkpoint
from remedi.training.classification_training import (
    FocalLoss,
    inverse_frequency_alpha,
)
from remedi.training.data.sample import yield_molecules_supervised_collate_fn
from remedi.training.data.samplers import lengths_from_ptr
from remedi.training.data.training_dataset import (
    TrainingMoleculeDataset,
    make_supervised_getitem,
)
from remedi.training.telemetry import TrainingTelemetry
from remedi_data.yaml_io import read_model, write_model
from remedi_eval.benchmark.metrics import (
    balanced_accuracy,
    macro_auroc_ovr,
    macro_f1,
)
from remedi_eval.protocol import load_benchmark

logger = logging.getLogger("remedi.chiral")
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a chiral-type classifier from scratch."
    )
    p.add_argument("--training_config", type=Path, required=True)
    p.add_argument("--dataset_path", type=Path, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument(
        "--device", type=str, default=None, help="Override compute device (e.g. cpu)."
    )
    return p.parse_args()


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    """Encoder + heads + pseudoscalar/geometric preprocessors. Excludes the
    frozen MACE foundation model living inside the preprocessor."""
    return (
        list(model.encoder.parameters())
        + list(model.multitask_heads.parameters())
        + list(model.preprocessor.atomic_preprocessor.parameters())
        + list(model.preprocessor.geometric_preprocessor.parameters())
    )


@torch.no_grad()
def _evaluate(model, loader, n_classes: int, max_batches: int | None):
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for bi, samples in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        labels_all.append(samples.regression_targets.cpu().numpy().reshape(-1))
        samples.to_(device)
        logits = model(samples).regression_predictions
        probs_all.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
    y_true = np.concatenate(labels_all).astype(int)
    probs = np.concatenate(probs_all, axis=0)
    return {
        "val_balanced_accuracy": balanced_accuracy(y_true, probs),
        "val_macro_f1": macro_f1(y_true, probs),
        "val_macro_auroc_ovr": macro_auroc_ovr(y_true, probs),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    global device
    if args.device is not None:
        device = args.device
    training_config = read_model(args.training_config, ClassifierTrainingConfig)
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    if args.max_epochs is not None:
        training_config.epochs = args.max_epochs
    architecture_config = read_model(
        training_config.model_config_path, ArchitectureConfig
    )
    if not isinstance(architecture_config, RegressionArchitectureConfig):
        raise ValueError(
            "chiral classifier requires a `kind: regression` architecture."
        )
    head_cfgs = architecture_config.regression_head_config
    if len(head_cfgs) != 1 or head_cfgs[0].n_classes is None:
        raise ValueError(
            "chiral classifier requires exactly one head with `n_classes` set."
        )
    n_classes = int(head_cfgs[0].n_classes)

    dataset_path = args.dataset_path or training_config.dataset_path
    benchmark = load_benchmark(dataset_path)
    full_dataset = benchmark.dataset
    labels = benchmark.class_indices()
    partition = benchmark.partition()
    train_idx, val_idx = partition.train, partition.valid
    logger.info(
        "stored split: train=%d valid=%d | train class counts=%s",
        len(train_idx),
        len(val_idx),
        np.bincount(labels[train_idx], minlength=n_classes).tolist(),
    )

    ds = TrainingMoleculeDataset(
        dataset_path, get_item=make_supervised_getitem(labels), in_memory=True
    )
    all_lengths = lengths_from_ptr(np.asarray(full_dataset.ptr[:]))
    train_loader = training_config.dataloader.build(
        Subset(ds, train_idx),
        lengths=all_lengths[train_idx],
        collate_fn=yield_molecules_supervised_collate_fn,
        shuffle=True,
        seed=training_config.seed,
    )
    val_loader = training_config.dataloader.build(
        Subset(ds, val_idx),
        lengths=all_lengths[val_idx],
        collate_fn=yield_molecules_supervised_collate_fn,
        shuffle=False,
        seed=training_config.seed,
    )

    out_base = training_config.create_run_directory()
    logger.info("training dir: %s", out_base)

    model = architecture_config.build()
    model.to(device)

    cls_cfg = training_config.classification
    alpha = (
        inverse_frequency_alpha(labels[train_idx], n_classes)
        if cls_cfg.class_balanced_alpha
        else None
    )
    criterion = FocalLoss(gamma=cls_cfg.focal_gamma, alpha=alpha).to(device)
    logger.info(
        "focal loss: gamma=%.2f alpha=%s",
        cls_cfg.focal_gamma,
        None if alpha is None else [round(float(a), 3) for a in alpha],
    )

    params = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        params,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_config.learning_rate,
        total_steps=training_config.epochs * max(1, len(train_loader)),
    )

    write_model(out_base / ARCHITECTURE_FILENAME, architecture_config)

    best_score = -np.inf
    with TrainingTelemetry(
        wandb_active=training_config.wandb_active,
        run_name=training_config.training_name,
        group_name=training_config.run_group,
        out_dir=out_base,
        config={
            "train_config": training_config.model_dump(),
            "architecture_config": architecture_config.model_dump(),
        },
    ) as telemetry:
        for epoch in range(training_config.epochs):
            model.train()
            running = 0.0
            n_seen = 0
            for bi, samples in enumerate(train_loader):
                if args.max_train_batches is not None and bi >= args.max_train_batches:
                    break
                targets = samples.regression_targets.to(device).long()
                samples.to_(device)
                logits = model(samples).regression_predictions
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, training_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                running += float(loss.detach()) * targets.numel()
                n_seen += int(targets.numel())

            train_loss = running / max(1, n_seen)
            metrics = _evaluate(model, val_loader, n_classes, args.max_val_batches)
            metrics.update(
                {
                    "epoch": epoch,
                    "train_focal_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            telemetry.log_epoch_metrics(metrics)
            logger.info(
                "epoch %d: train_focal=%.4f val_bal_acc=%.4f val_macro_f1=%.4f "
                "val_macro_auroc=%.4f",
                epoch,
                train_loss,
                metrics["val_balanced_accuracy"],
                metrics["val_macro_f1"],
                metrics["val_macro_auroc_ovr"],
            )

            score = metrics["val_balanced_accuracy"]
            if score > best_score:
                best_score = score
                save_checkpoint(model, out_base)
                logger.info(
                    "  new best val balanced accuracy %.4f -> checkpointed", score
                )

    logger.info("done. best val balanced accuracy: %.4f", best_score)


if __name__ == "__main__":
    main()
