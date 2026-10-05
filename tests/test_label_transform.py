"""Per-label transforms declared in ``dataset.yaml`` and applied by the harness."""

from __future__ import annotations

from pathlib import Path
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _spec(labels: list[LabelColumn]) -> DatasetSpec:
    return DatasetSpec(dataset_id="toy", labels=labels, source_kind="test")


def _yaml_round_trip(spec: DatasetSpec) -> tuple[dict, DatasetSpec]:
    text = yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True))
    raw = yaml.safe_load(text)
    return raw, DatasetSpec.model_validate(raw)


def test_identity_is_default_and_not_serialised() -> None:
    label = LabelColumn(name="y", task_type=TaskType.regression)
    assert isinstance(label.transform, IdentityLabelTransform)
    raw, reloaded = _yaml_round_trip(_spec([label]))
    assert raw["labels"] == [{"name": "y", "task_type": "regression"}]
    assert reloaded.labels[0] == label


def test_explicit_identity_round_trips() -> None:
    raw = {"name": "y", "task_type": "regression", "transform": {"kind": "identity"}}
    label = LabelColumn.model_validate(raw)
    assert isinstance(label.transform, IdentityLabelTransform)
    assert "transform" not in label.model_dump(mode="json")


def test_log10_yaml_round_trip() -> None:
    label = LabelColumn(
        name="potency",
        task_type=TaskType.regression,
        transform=Log10LabelTransform(clip_minimum=0.5, offset=2.0),
    )
    raw, reloaded = _yaml_round_trip(_spec([label]))
    assert raw["labels"][0]["transform"] == {
        "kind": "log10",
        "clip_minimum": 0.5,
        "offset": 2.0,
    }
    assert reloaded.labels[0] == label
    assert isinstance(reloaded.labels[0].transform, Log10LabelTransform)


def test_log10_from_minimal_yaml_uses_competition_defaults() -> None:
    raw = yaml.safe_load("name: y\ntask_type: regression\ntransform:\n  kind: log10\n")
    transform = LabelColumn.model_validate(raw).transform
    assert transform == Log10LabelTransform(clip_minimum=0.0, offset=1.0)


def test_with_structures_keeps_transform() -> None:
    label = LabelColumn(
        name="y", task_type=TaskType.regression, transform=Log10LabelTransform()
    )
    dataset_spec = _spec([label]).with_structures("etkdg_mmff")
    assert isinstance(dataset_spec.labels[0].transform, Log10LabelTransform)


def test_log10_known_values_clips_negatives_and_keeps_nan() -> None:
    values = np.array([-5.0, 0.0, 9.0, 99.0, np.nan])
    result = Log10LabelTransform().apply(values)
    expected = np.log10(np.clip(values, a_min=0, a_max=None) + 1)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    np.testing.assert_allclose(result[:4], [0.0, 0.0, 1.0, 2.0])
    assert np.isnan(result[4])


def test_identity_apply_keeps_values_and_nan() -> None:
    values = np.array([-1.0, np.nan, 3.0])
    np.testing.assert_array_equal(
        IdentityLabelTransform().apply(values), values, strict=False
    )


@pytest.mark.parametrize("task_type", [TaskType.classification, TaskType.multiclass])
def test_log10_rejected_on_non_regression_label(task_type: TaskType) -> None:
    extra = {"n_classes": 3} if task_type is TaskType.multiclass else {}
    with pytest.raises(ValidationError, match="only allowed on regression"):
        LabelColumn(
            name="y",
            task_type=task_type,
            transform=Log10LabelTransform(),
            **extra,
        )


@pytest.mark.parametrize(
    ("clip_minimum", "offset"), [(0.0, 0.0), (-1.0, 1.0), (-2.0, 0.5)]
)
def test_log10_rejects_non_positive_argument(
    clip_minimum: float, offset: float
) -> None:
    with pytest.raises(ValidationError, match="clip_minimum \\+ offset > 0"):
        Log10LabelTransform(clip_minimum=clip_minimum, offset=offset)


def test_unknown_transform_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        LabelColumn.model_validate(
            {"name": "y", "task_type": "regression", "transform": {"kind": "sqrt"}}
        )


def test_apply_label_transforms_per_column() -> None:
    spec = _spec(
        [
            LabelColumn(name="raw", task_type=TaskType.regression),
            LabelColumn(
                name="logged",
                task_type=TaskType.regression,
                transform=Log10LabelTransform(),
            ),
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


def test_apply_label_transforms_rejects_unknown_column() -> None:
    spec = _spec([LabelColumn(name="y", task_type=TaskType.regression)])
    with pytest.raises(ValueError, match="not a label"):
        apply_label_transforms(np.zeros((2, 1)), ["other"], spec)


def test_apply_label_transforms_rejects_shape_mismatch() -> None:
    spec = _spec([LabelColumn(name="y", task_type=TaskType.regression)])
    with pytest.raises(ValueError, match="do not match"):
        apply_label_transforms(np.zeros((2, 2)), ["y"], spec)


def test_build_targets_masks_then_transforms() -> None:
    spec = _spec(
        [
            LabelColumn(
                name="y", task_type=TaskType.regression, transform=Log10LabelTransform()
            )
        ]
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


def _committed_spec_paths() -> list[Path]:
    benchmark_data = REPOSITORY_ROOT / "benchmark_data"
    return sorted(
        [
            *benchmark_data.glob("bundles/*/dataset.yaml"),
            *benchmark_data.glob("datasets/*/dataset.yaml"),
        ]
    )


@pytest.mark.parametrize(
    "spec_path", _committed_spec_paths(), ids=lambda path: path.parent.name
)
def test_committed_specs_load_and_redump_unchanged(spec_path: Path) -> None:
    raw = yaml.safe_load(spec_path.read_text())
    spec = DatasetSpec.model_validate(raw)
    # Dumped as bundle.py's yaml writer dumps it: identical to what is on disk.
    assert spec.model_dump(mode="json", exclude_none=True) == raw
