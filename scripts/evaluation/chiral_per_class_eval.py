"""Per-class diagnostics for the trained chiral-type classifiers.

Training only logged aggregate balanced-accuracy / macro-F1 / macro-AUROC. This
reloads each ablation checkpoint (the *actual* trained encoder + preprocessors +
classification head, not a re-fit probe) and runs it over the stored validation
split to recover the per-class precision / recall / F1 / support and the
confusion matrix that the headline numbers hide.

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

import numpy as np
import pydantic_yaml as pyaml
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import Subset

from remedi.configuration.architecture_config import ArchitectureConfig
from remedi.configuration.training_config import TrainingConfig
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import Split
from remedi.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    make_supervised_getitem,
)
from remedi.data_handling.sample import yield_molecules_supervised_collate_fn
from remedi.training.data.samplers import lengths_from_ptr

logger = logging.getLogger("remedi.chiral.per_class")
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["achiral", "central", "axial", "helical", "planar"]

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
    arch = pyaml.parse_yaml_file_as(
        ArchitectureConfig, run_dir / "post_training_architecture_config.yaml"
    )
    model = arch.build()
    model.to(device)
    ld = lambda f: torch.load(run_dir / f, map_location=device)  # noqa: E731
    model.encoder.load_state_dict(ld("encoder.pth"))
    model.preprocessor.atomic_preprocessor.load_state_dict(
        ld("atomic_preprocessor.pth")
    )
    model.preprocessor.geometric_preprocessor.load_state_dict(
        ld("geometric_preprocessor.pth")
    )
    model.multitask_heads.load_state_dict(ld("classification_head.pth"))
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


def _split_indices(ds: MoleculeDataset, split: Split) -> np.ndarray:
    codes = np.asarray(ds.split[:], dtype=np.uint8)
    return np.nonzero(codes == split.value)[0]


def evaluate_run(run_dir: Path, cfg_dir: Path, split: Split, n_classes: int) -> dict:
    training_config = pyaml.parse_yaml_file_as(
        TrainingConfig, cfg_dir / "training_config.yaml"
    )
    dataset_path = training_config.dataset_path
    full = MoleculeDataset.open_existing_dataset_from_dir(dataset_path)
    labels = np.asarray(full.targets_system[:, 0]).astype(np.int64)
    idx = _split_indices(full, split)

    ds = TrainingMoleculeDataset(
        dataset_path, get_item=make_supervised_getitem(labels), in_memory=True
    )
    all_lengths = lengths_from_ptr(np.asarray(full.ptr[:]))
    loader = training_config.dataloader.build(
        Subset(ds, idx),
        lengths=all_lengths[idx],
        collate_fn=yield_molecules_supervised_collate_fn,
        shuffle=False,
    )

    model = _load_model(run_dir)
    y_true, y_pred = _predict(model, loader)

    labels_arr = np.arange(n_classes)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_arr, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_arr)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "balanced_accuracy": float(bal_acc),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
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
    print("confusion matrix (rows=true, cols=pred), order:", CLASS_NAMES)
    for cname, row in zip(CLASS_NAMES, res["confusion_matrix"], strict=True):
        print(f"  {cname:9s} {row}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", type=Path, required=True)
    p.add_argument("--configs_dir", type=Path, required=True)
    p.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    p.add_argument("--out", type=Path, default=None, help="optional JSON dump")
    p.add_argument("--n_classes", type=int, default=5)
    args = p.parse_args()

    split = Split.valid if args.split == "valid" else Split.test
    all_res: dict[str, dict] = {}
    for suffix, cfg_name in RUNS.items():
        run_dir = _find_run_dir(args.ablation_dir, suffix)
        cfg_dir = args.configs_dir / cfg_name
        logger.info("evaluating %s (ckpt=%s)", cfg_name, run_dir.name)
        res = evaluate_run(run_dir, cfg_dir, split, args.n_classes)
        all_res[cfg_name] = res
        _print_run(cfg_name, res)

    if args.out is not None:
        args.out.write_text(json.dumps(all_res, indent=2))
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
