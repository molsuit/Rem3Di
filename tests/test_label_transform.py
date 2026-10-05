"""Per-label transforms declared in ``dataset.yaml`` and applied by the harness."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from remedi.data_handling.bundle.spec import (
    DatasetSpec,
    IdentityLabelTransform,
    LabelColumn,
    Log10LabelTransform,
)
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.runner import (
    _build_targets_with_nan,
    apply_label_transforms,
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


def test_build_targets_masks_then_transforms() -> None:
    spec = _spec(
        [LabelColumn(name="y", task_type=TaskType.regression, transform=LOG10)]
    )
    fake_dataset = SimpleNamespace(
        targets_system=np.array([[9.0], [-4.0], [123.0]]),
        mask_system=np.array([[1], [1], [0]]),
        config=SimpleNamespace(
            tasks=SimpleNamespace(system_cols=[SimpleNamespace(name="y")])
        ),
    )
    targets = _build_targets_with_nan(fake_dataset, spec)  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(targets, [[1.0], [0.0], [np.nan]], equal_nan=True)
