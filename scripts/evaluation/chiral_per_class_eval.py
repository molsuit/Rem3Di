"""Per-class diagnostics for the trained chiral-type classifiers.

Training only logged aggregate balanced-accuracy / macro-F1 / macro-AUROC. This
reloads each ablation checkpoint (the *actual* trained encoder + preprocessors +
classification head, not a re-fit probe) and runs it over the stored validation
split to recover the per-class precision / recall / F1 / support and the
confusion matrix that the headline numbers hide.

The per-class table and the confusion matrix come from
:func:`remedi_eval.benchmark.metrics.multiclass_report`, headed with the class
names the dataset spec declares (``LabelColumn.class_names``); the scored rows
are a fold of one split column of the dataset table
(:meth:`remedi_eval.protocol.Benchmark.partition`).

Run on a GPU node (MACE featurizes on the fly):
    uv run python scripts/evaluation/chiral_per_class_eval.py \
        --ablation_dir /path/to/training_runs/chiral_cat_ablation \
        --configs_dir configs/training/chiral_cat_ablation \
        --split valid
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Subset

from remedi.configuration.architecture_config import ArchitectureConfig
from remedi.configuration.training_config import ClassifierTrainingConfig
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, load_checkpoint
from remedi.training.data.training_dataset import (
    TrainingMoleculeDataset,
    make_supervised_getitem,
)
from remedi_data.logging_setup import setup_logging
from remedi_data.yaml_io import read_model
from remedi_eval.benchmark.cells import multiclass_labelling
from remedi_eval.benchmark.metrics import balanced_accuracy, multiclass_report
from remedi_eval.protocol import load_benchmark

logger = logging.getLogger("remedi.chiral.per_class")
device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint-dir suffix -> config-dir name
RUNS = {
    "chiral_ps_focal": "ps_focal",
    "chiral_nops_focal": "nops_focal",
    "chiral_ps_ce": "ps_ce",
    "chiral_nops_ce": "nops_ce",
}


def _find_run_dir(ablation_dir: Path, suffix: str) -> Path:
    matches = sorted(ablation_dir.glob(f"*-{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"no checkpoint dir matching *-{suffix} under {ablation_dir}"
        )
    return matches[-1]


def _load_model(run_dir: Path):
    arch = read_model(run_dir / ARCHITECTURE_FILENAME, ArchitectureConfig)
    model = arch.build()
    model.to(device)
    load_checkpoint(model, run_dir, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def _predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for samples in loader:
        y_true.append(samples.regression_targets.cpu().numpy().reshape(-1))
        samples.to_(device)
        logits = model(samples).regression_predictions
        y_pred.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
    return np.concatenate(y_true).astype(int), np.concatenate(y_pred).astype(int)


def evaluate_run(
    run_dir: Path,
    cfg_dir: Path,
    split_column: str,
    fold: Literal["valid", "test"],
) -> dict:
    training_config = read_model(
        cfg_dir / "training_config.yaml", ClassifierTrainingConfig
    )
    dataset_path = training_config.dataset_path
    benchmark = load_benchmark(dataset_path)
    n_classes, class_names = multiclass_labelling(benchmark.spec)
    if n_classes is None:
        raise ValueError(f"{dataset_path} declares no single multiclass label column")
    labels = benchmark.class_indices()
    partition = benchmark.partition(split_column)
    idx = partition.valid if fold == "valid" else partition.test

    ds = TrainingMoleculeDataset(dataset_path, get_item=make_supervised_getitem(labels))
    all_lengths = benchmark.dataset.structures.atom_counts()
    model = _load_model(run_dir)
    loader = training_config.dataloader.build(
        Subset(ds, idx),
        lengths=all_lengths[idx],
        collate_fn=model.preprocessor.molecule_collator(with_regression_targets=True),
        shuffle=False,
    )
    y_true, y_pred = _predict(model, loader)

    # ``y_pred`` holds argmax labels, which multiclass_report takes as-is.
    report = multiclass_report(y_true, y_pred, n_classes, class_names)
    per_class = report.per_class.set_index("class_name")
    return {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "class_names": report.class_names,
        "per_class": {
            str(name): {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "support": int(row["support"]),
            }
            for name, row in per_class.iterrows()
        },
        "confusion_matrix": report.confusion_matrix.tolist(),
    }


def _print_run(name: str, res: dict) -> None:
    print(
        f"\n===== {name}  (val balanced accuracy = {res['balanced_accuracy']:.4f}) ====="
    )
    print(f"{'class':9s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'support':>8s}")
    for cname, m in res["per_class"].items():
        print(
            f"{cname:9s} {m['precision']:6.3f} {m['recall']:7.3f} "
            f"{m['f1']:6.3f} {m['support']:8d}"
        )
    print("confusion matrix (rows=true, cols=pred), order:", res["class_names"])
    for cname, row in zip(res["class_names"], res["confusion_matrix"], strict=True):
        print(f"  {cname:9s} {row}")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", type=Path, required=True)
    p.add_argument("--configs_dir", type=Path, required=True)
    p.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    p.add_argument(
        "--split_column",
        type=str,
        default="split",
        help="split column of the dataset table the fold is taken from",
    )
    p.add_argument("--out", type=Path, default=None, help="optional JSON dump")
    args = p.parse_args()

    all_res: dict[str, dict] = {}
    for suffix, cfg_name in RUNS.items():
        run_dir = _find_run_dir(args.ablation_dir, suffix)
        cfg_dir = args.configs_dir / cfg_name
        logger.info("evaluating %s (ckpt=%s)", cfg_name, run_dir.name)
        res = evaluate_run(run_dir, cfg_dir, args.split_column, args.split)
        all_res[cfg_name] = res
        _print_run(cfg_name, res)

    if args.out is not None:
        args.out.write_text(json.dumps(all_res, indent=2))
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
