"""Chiral-type classification report — the per-class diagnostic the flat
benchmark leaderboard can't carry.

``benchmark_panel`` scores the ``chiral_cat`` multiclass task with a single
headline number (balanced accuracy) per descriptor x learner. That hides *which*
chirality classes the pseudoscalar embedding actually separates — the
interesting question, given the dataset is dominated by ``achiral`` / ``central``
with only 37 ``helical`` and 59 ``planar`` molecules. This task fits one probe on
the run's frozen embedding and emits the full picture: per-class
precision / recall / F1 / support, the confusion matrix (counts + a heatmap), and
the headline balanced-accuracy / macro-F1 / macro-OvR-AUROC together.

It reuses the shared machinery — ``EmbeddingSpec`` for the cached descriptor
matrix, ``_split_masks`` / ``_build_targets_with_nan`` for the stored split, and
``LearnerConfig`` for the probe — so it stays consistent with the panel. The
default probe is a focal-loss MLP, which is the right objective for this
imbalance.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.benchmark.learners import (
    LearnerConfig,
    MlpLearnerConfig,
)
from remedi.evaluation.benchmark.metrics import (
    balanced_accuracy,
    macro_auroc_ovr,
    macro_f1,
)
from remedi.evaluation.benchmark.runner import (
    _build_targets_with_nan,
    _split_masks,
)
from remedi.evaluation.framework.context import EvalContext
from remedi.evaluation.framework.resources import EmbeddingSpec
from remedi.evaluation.results import (
    ArrayResult,
    EvalResult,
    FigureResult,
    PydanticResult,
    TableResult,
)

logger = logging.getLogger(__name__)


class ChiralReportSummary(BaseModel):
    dataset_id: str
    descriptor_name: str
    learner_kind: str
    n_classes: int
    n_train: int
    n_val: int
    n_test: int
    balanced_accuracy: float
    macro_f1: float
    macro_auroc_ovr: float


class ChiralReportConfig(BaseModel):
    """Per-class confusion-matrix report for the chiral-type classification."""

    kind: Literal["chiral_report"] = "chiral_report"
    # Directory of the prepared chiral_cat zarr (built by build_chiral_cat.py).
    zarr_path: Path
    dataset_id: str = "chiral_cat"
    # Probe fit on the frozen embedding. Focal-loss MLP by default — the right
    # objective for the dominant-class imbalance.
    learner: LearnerConfig = Field(
        default_factory=lambda: MlpLearnerConfig(loss="focal")
    )
    # Display names for classes 0..4 (paper ordering); used to label the
    # per-class table and confusion matrix.
    class_names: list[str] = Field(
        default_factory=lambda: ["achiral", "central", "axial", "helical", "planar"]
    )

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        out_rel = Path("chiral_report")
        dataset = MoleculeDataset.open_existing_dataset_from_dir(self.zarr_path)
        X = ctx.resources.get(
            EmbeddingSpec(
                dataset_id=self.dataset_id,
                descriptor=ctx.model,
                dataset=dataset,
                cache_dir=ctx.resource_cache_dir,
            )
        )
        y = _build_targets_with_nan(dataset)[:, 0]
        tr, va, te = _split_masks(dataset, None)
        n_classes = int(np.nanmax(y)) + 1

        probs = self.learner.build().fit_predict_multiclass(
            X[tr], y[tr], X[va], y[va], X[te], n_classes, ctx.seed
        )
        y_true = y[te].astype(int)
        y_pred = probs.argmax(axis=1)

        labels = np.arange(n_classes)
        names = self._labels(n_classes)

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_class = pd.DataFrame(
            {
                "class_id": labels,
                "class_name": names,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_frame = pd.DataFrame(cm, index=names, columns=names)
        cm_frame.insert(0, "true\\pred", names)

        learner_kind = str(self.learner.learner_kind)  # type: ignore[union-attr]
        summary = ChiralReportSummary(
            dataset_id=self.dataset_id,
            descriptor_name=ctx.model.name,
            learner_kind=learner_kind,
            n_classes=n_classes,
            n_train=int(tr.sum()),
            n_val=int(va.sum()),
            n_test=int(te.sum()),
            balanced_accuracy=balanced_accuracy(y_true, probs),
            macro_f1=macro_f1(y_true, probs),
            macro_auroc_ovr=macro_auroc_ovr(y_true, probs),
        )
        logger.info(
            "chiral_report %s/%s: balanced_acc=%.4f macro_f1=%.4f",
            ctx.model.name,
            learner_kind,
            summary.balanced_accuracy,
            summary.macro_f1,
        )

        yield PydanticResult(file_name=out_rel / "summary.yaml", obj=summary)
        yield TableResult(file_name=out_rel / "per_class.csv", frame=per_class)
        yield TableResult(file_name=out_rel / "confusion_matrix.csv", frame=cm_frame)
        yield ArrayResult(
            file_name=out_rel / "confusion_matrix.npz",
            arrays={"confusion_matrix": cm, "labels": labels},
        )
        yield FigureResult(
            file_name=out_rel / "confusion_matrix.png",
            figure=_confusion_heatmap(cm, names, ctx.model.name),
            save_kwargs={"dpi": 150, "bbox_inches": "tight"},
        )

    def _labels(self, n_classes: int) -> list[str]:
        if len(self.class_names) >= n_classes:
            return self.class_names[:n_classes]
        return self.class_names + [
            f"class_{i}" for i in range(len(self.class_names), n_classes)
        ]


def _confusion_heatmap(cm: np.ndarray, names: list[str], title: str):
    """Row-normalized confusion-matrix heatmap (recall per true class)."""
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )

    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 1.5, 1.4 * len(names) + 1.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Chiral-type confusion (row-normalized)\n{title}")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig
