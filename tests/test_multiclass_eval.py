"""Multiclass support across the benchmark framework.

Covers:
  * Each learner's ``fit_predict_multiclass`` returns ``(n_test, n_classes)``
    rows that sum to ~1 (the focal-loss MLP path as a CPU-gated opt-in), and a
    class absent from train still keeps its column.
  * The three multiclass metrics behave on perfect / chance predictions, and the
    per-class report keeps a row for every declared class.
  * The panel task scores a multiclass benchmark as one row and writes the
    per-class report per cell, with the spec's class names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from remedi.data_handling.bundle import EvalMetric, LabelColumn
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.descriptors import EcfpConfig
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
    multiclass_report,
)
from remedi.evaluation.framework import BenchmarkPanelConfig, EvalManifest, run_manifest
from remedi.evaluation.results import ArrayResult

from .helpers.bundle_fixtures import (
    ACHIRAL_TEN_SMILES,
    TEN_ROW_SPLIT,
    write_small_dataset,
)

N_CLASSES = 5


def _synthetic_multiclass(n_per_class: int, dimension: int, seed: int = 0):
    """Gaussian blobs, one per class, separated along distinct axes."""
    generator = np.random.default_rng(seed)
    features, labels = [], []
    for class_index in range(N_CLASSES):
        center = np.zeros(dimension)
        center[class_index % dimension] = 4.0
        features.append(generator.standard_normal((n_per_class, dimension)) + center)
        labels.append(np.full(n_per_class, class_index))
    permutation = generator.permutation(n_per_class * N_CLASSES)
    labels_array = np.concatenate(labels).astype(float)
    return np.vstack(features)[permutation], labels_array[permutation]


def _fit_predict(learner, features, labels, n_train: int, n_valid: int) -> np.ndarray:
    """Train on the first ``n_train`` rows, validate on the next ``n_valid``."""
    end = n_train + n_valid
    return learner.fit_predict_multiclass(
        features[:n_train],
        labels[:n_train],
        features[n_train:end],
        labels[n_train:end],
        features[end:],
        N_CLASSES,
    )


@pytest.mark.parametrize(
    ("learner", "minimum_accuracy"),
    [
        (LinearLearner(), 0.8),
        (LightGBMLearner(n_estimators=50), None),
        (
            MlpLearner(
                hidden_dims=(32,),
                max_epochs=30,
                device="cpu",
                allow_cpu_mlp=True,
                loss="focal",
                focal_gamma=2.0,
            ),
            None,
        ),
        (NullLearner(), None),
    ],
    ids=["linear", "lightgbm", "mlp_focal", "null"],
)
def test_fit_predict_multiclass_returns_probabilities(
    learner, minimum_accuracy: float | None
) -> None:
    features, labels = _synthetic_multiclass(40, 6)
    probabilities = _fit_predict(learner, features, labels, 120, 40)
    test_labels = labels[160:]
    n_test = len(test_labels)
    assert probabilities.shape == (n_test, N_CLASSES)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(n_test), atol=1e-5)
    if isinstance(learner, NullLearner):
        # Every test row gets the identical train base-rate vector.
        assert np.allclose(probabilities, probabilities[0])
    if minimum_accuracy is not None:
        accuracy = (probabilities.argmax(1) == test_labels.astype(int)).mean()
        assert accuracy > minimum_accuracy


def test_missing_class_in_train_still_maps_to_its_column() -> None:
    # Train has classes {0,1,2,3}; class 4 absent. The proba block must still be
    # 5-wide with an all-zero column 4.
    features, labels = _synthetic_multiclass(30, 6)
    keep = labels < 4
    probabilities = _fit_predict(LinearLearner(), features[keep], labels[keep], 80, 20)
    assert probabilities.shape[1] == N_CLASSES
    assert np.allclose(probabilities[:, 4], 0.0)


def test_multiclass_metrics_perfect_and_chance() -> None:
    y_true = np.array([0, 1, 2, 3, 4])
    perfect = np.eye(5)[y_true]
    assert balanced_accuracy(y_true, perfect) == 1.0
    assert macro_f1(y_true, perfect) == 1.0
    assert macro_auroc_ovr(y_true, perfect) == 1.0
    # A single test class -> OvR AUROC undefined -> NaN.
    single_class = np.zeros(4, dtype=int)
    assert np.isnan(macro_auroc_ovr(single_class, np.eye(5)[single_class]))


def test_multiclass_report_shape_and_fallback_names() -> None:
    """Fixed label order, one row per declared class, generated names by default."""
    y_true = np.array([0, 0, 1, 1])
    # Class 2 never appears in the fold: it keeps its row and its column.
    probabilities = np.eye(3)[[0, 1, 1, 1]]

    report = multiclass_report(y_true, probabilities, 3)
    assert list(report.per_class["class_name"]) == ["class_0", "class_1", "class_2"]
    assert list(report.per_class["support"]) == [2, 2, 0]
    assert int(report.confusion_matrix.sum()) == len(y_true)
    # zero_division=0 rather than a raise for the class with no support.
    assert report.per_class.loc[2, "f1"] == 0.0

    named = multiclass_report(y_true, probabilities, 3, ["a", "b", "c"])
    assert named.class_names == ["a", "b", "c"]


def test_panel_task_scores_multiclass_and_writes_the_per_class_report(
    tmp_path: Path,
) -> None:
    """One scored row per cell, and the cell artifacts land under the benchmark's
    own directory, headed with the class names the dataset spec declares."""
    n_rows = len(ACHIRAL_TEN_SMILES)
    n_classes = 3
    class_names = ["achiral", "central", "axial"]
    write_small_dataset(
        tmp_path / "datasets",
        dataset_id="toy_multiclass",
        labels=[
            LabelColumn(
                name="chirality_type",
                task_type=TaskType.multiclass,
                n_classes=n_classes,
                class_names=class_names,
            )
        ],
        metrics=[EvalMetric.balanced_accuracy],
        targets=(np.arange(n_rows) % n_classes).astype(float),
    )
    manifest = EvalManifest(
        model=EcfpConfig(name="ecfp_256", length=256),
        output_root=tmp_path / "eval_out" / "model_x",
        tasks=[
            BenchmarkPanelConfig(
                eval_root=tmp_path / "datasets",
                learners=[LinearLearnerConfig(ridge_alpha=1.0)],
            )
        ],
    )
    assert run_manifest(manifest).n_failed == 0

    benchmark_dir = manifest.output_root / "benchmark"
    results = pd.read_csv(benchmark_dir / "results.csv")
    assert len(results) == 1
    row = results.iloc[0]
    assert pd.isna(row["target_col"])
    assert row["metric_name"] == EvalMetric.balanced_accuracy.value
    assert 0.0 <= row["metric_value"] <= 1.0

    cell_dir = benchmark_dir / "toy_multiclass" / "ecfp_256__linear"
    per_class = pd.read_csv(cell_dir / "per_class.csv")
    assert list(per_class["class_name"]) == class_names

    confusion = pd.read_csv(cell_dir / "confusion_matrix.csv")
    assert list(confusion.columns) == ["true_class", *class_names]
    assert list(confusion["true_class"]) == class_names
    # The fixture's 6/2/2 partition: the whole test fold is accounted for.
    n_test = sum(label == "test" for label in TEN_ROW_SPLIT)
    assert row["n_test"] == n_test
    assert confusion[class_names].to_numpy().sum() == n_test

    arrays = ArrayResult.load(cell_dir / "confusion_matrix.npz")
    assert arrays["confusion_matrix"].shape == (n_classes, n_classes)
    assert [str(label) for label in arrays["labels"]] == class_names
