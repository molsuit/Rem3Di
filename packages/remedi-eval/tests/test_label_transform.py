"""Per-label transforms declared in ``dataset.yaml`` and applied by the harness."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from remedi_data.bundle import EvalMetric
from remedi_data.bundle.spec import (
    DatasetSpec,
    IdentityLabelTransform,
    LabelColumn,
    Log10LabelTransform,
)
from remedi_data.dataset.tasks import TaskType
from remedi_data.testing import write_small_dataset
from remedi_eval.protocol import (
    apply_label_transforms,
    build_targets_with_nan,
    load_benchmark,
)

LOG10 = Log10LabelTransform()


def _spec(labels: list[LabelColumn]) -> DatasetSpec:
    return DatasetSpec(dataset_id="toy", labels=labels, source_kind="test")


@pytest.mark.parametrize(
    ("transform_yaml", "dumped_transform"),
    [
        # The identity default is left out of the dump, given or not.
        ("", None),
        ("transform:\n  kind: identity\n", None),
        # A minimal log10 gets the competition defaults, written out in full.
        (
            "transform:\n  kind: log10\n",
            {"kind": "log10", "clip_minimum": 0.0, "offset": 1.0},
        ),
        (
            "transform: {kind: log10, clip_minimum: 0.5, offset: 2.0}\n",
            {"kind": "log10", "clip_minimum": 0.5, "offset": 2.0},
        ),
    ],
    ids=["default", "explicit_identity", "log10_defaults", "log10_explicit"],
)
def test_transform_yaml_round_trip(
    transform_yaml: str, dumped_transform: dict | None
) -> None:
    raw = yaml.safe_load("name: y\ntask_type: regression\n" + transform_yaml)
    spec = _spec([LabelColumn.model_validate(raw)])
    dumped = yaml.safe_load(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True))
    )
    expected = {"name": "y", "task_type": "regression"}
    if dumped_transform is not None:
        expected["transform"] = dumped_transform
    assert dumped["labels"] == [expected]
    assert DatasetSpec.model_validate(dumped) == spec


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        # Negatives clip to zero, NaN (a missing label) stays NaN.
        (LOG10, [0.0, 0.0, 1.0, 2.0, np.nan]),
        (IdentityLabelTransform(), [-5.0, 0.0, 9.0, 99.0, np.nan]),
    ],
)
def test_transform_apply(
    transform: IdentityLabelTransform | Log10LabelTransform, expected: list[float]
) -> None:
    values = np.array([-5.0, 0.0, 9.0, 99.0, np.nan])
    np.testing.assert_allclose(transform.apply(values), expected, equal_nan=True)


@pytest.mark.parametrize("task_type", [TaskType.classification, TaskType.multiclass])
def test_log10_rejected_on_non_regression_label(task_type: TaskType) -> None:
    extra = {"n_classes": 3} if task_type is TaskType.multiclass else {}
    with pytest.raises(ValidationError, match="only allowed on regression"):
        LabelColumn(name="y", task_type=task_type, transform=LOG10, **extra)


@pytest.mark.parametrize(
    ("clip_minimum", "offset"), [(0.0, 0.0), (-1.0, 1.0), (-2.0, 0.5)]
)
def test_log10_rejects_non_positive_argument(
    clip_minimum: float, offset: float
) -> None:
    with pytest.raises(ValidationError, match="clip_minimum \\+ offset > 0"):
        Log10LabelTransform(clip_minimum=clip_minimum, offset=offset)


def test_apply_label_transforms_per_column() -> None:
    spec = _spec(
        [
            LabelColumn(name="raw", task_type=TaskType.regression),
            LabelColumn(name="logged", task_type=TaskType.regression, transform=LOG10),
        ]
    )
    targets = np.array([[1.0, 9.0], [np.nan, -3.0], [2.0, np.nan]])
    # Column order of the zarr need not match the spec's label order.
    result = apply_label_transforms(targets[:, ::-1], ["logged", "raw"], spec)
    np.testing.assert_allclose(
        result, [[1.0, 1.0], [0.0, np.nan], [np.nan, 2.0]], equal_nan=True
    )
    # The input is not modified in place.
    assert targets[0, 1] == 9.0


@pytest.mark.parametrize(
    ("shape", "column_names", "message"),
    [((2, 1), ["other"], "not a label"), ((2, 2), ["y"], "do not match")],
)
def test_apply_label_transforms_rejects(
    shape: tuple[int, int], column_names: list[str], message: str
) -> None:
    spec = _spec([LabelColumn(name="y", task_type=TaskType.regression)])
    with pytest.raises(ValueError, match=message):
        apply_label_transforms(np.zeros(shape), column_names, spec)


def test_build_targets_reads_the_label_columns_then_transforms() -> None:
    spec = _spec(
        [
            LabelColumn(name="y", task_type=TaskType.regression, transform=LOG10),
            LabelColumn(name="z", task_type=TaskType.regression),
        ]
    )
    table = pd.DataFrame(
        {"z": [1.0, 2.0, 3.0], "other": [0.0, 0.0, 0.0], "y": [9.0, -4.0, np.nan]}
    )
    targets = build_targets_with_nan(table, spec)
    np.testing.assert_allclose(
        targets, [[1.0, 1.0], [0.0, 2.0], [np.nan, 3.0]], equal_nan=True
    )


def test_a_benchmark_reads_its_labels_from_the_spec_and_the_table(
    tmp_path: Path,
) -> None:
    """The zarr holds no label arrays and no label names; the protocol takes the
    names and task types from ``dataset.yaml`` and the values from the table."""
    labels = [
        LabelColumn(name="active", task_type=TaskType.classification),
        LabelColumn(name="toxic", task_type=TaskType.classification),
    ]
    targets = np.tile([[1.0, 0.0], [0.0, np.nan]], (5, 1))
    directory = write_small_dataset(
        tmp_path,
        dataset_id="two_labels",
        labels=labels,
        metrics=[EvalMetric.macro_auroc],
        targets=targets,
    )
    shutil.rmtree(directory / "tasks", ignore_errors=True)
    config_path = directory / "dataset_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.pop("tasks", None)
    config_path.write_text(yaml.safe_dump(config))

    benchmark = load_benchmark(directory)

    assert benchmark.task_kind == "multilabel"
    assert benchmark.target_columns == ["active", "toxic"]
    np.testing.assert_array_equal(benchmark.targets(), targets)
