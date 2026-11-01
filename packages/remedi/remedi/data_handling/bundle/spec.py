"""``dataset.yaml``: the :class:`DatasetSpec` of a bundle or a dataset.

See ``BENCHMARK_DATA_FORMAT.md`` §10.2. The same spec heads two kinds of
directory: a **bundle** (a table with one row per stereoisomer, before any
structures exist; ``geometry_origin`` unset) and a **dataset** (a zarr with the
table beside it, one row per structure; ``geometry_origin`` set). Labels are
top level, because a pretraining corpus has labels too; everything only scoring
needs lives in the optional ``evaluation`` block.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from remedi.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)

#: The identity columns every table starts with, in order.
IDENTITY_COLUMNS: tuple[str, ...] = ("stereoisomer_id", "molecule_id", "enantiomer_of")
#: Present iff the spec declares ``smiles: true``.
SMILES_COLUMNS: tuple[str, ...] = ("isomeric_smiles", "nonisomeric_smiles")
#: The per-structure physics a dataset table carries and the zarr stores.
CHARGE_COLUMNS: tuple[str, ...] = ("total_charge", "multiplicity")

#: The only values a split column may hold (invariant 5).
SPLIT_VALUES: frozenset[str] = frozenset({"train", "valid", "test", "unassigned"})

#: Where a dataset's coordinates came from.
GeometryOrigin = Literal["source", "etkdg_mmff"]

#: The identity level a split column must be constant within (invariant 6).
SplitGroup = Literal["molecule_id", "stereoisomer_id"]


class EvalMetric(StrEnum):
    """Metrics an evaluation may ask for. Values are the strings written to yaml."""

    rmse = "RMSE"
    mae = "MAE"
    r2 = "R2"
    spearman = "Spearman"
    auroc = "AUROC"
    auprc = "AUPRC"
    macro_auroc = "macro-AUROC"
    balanced_accuracy = "balanced-accuracy"
    macro_f1 = "macro-F1"
    macro_auroc_ovr = "macro-AUROC-OvR"
    pair_ranking_accuracy = "pair-ranking-accuracy"


#: The metrics computed alongside the headline one, per task type (§7e).
METRICS_BY_TASK_TYPE: dict[TaskType, tuple[EvalMetric, ...]] = {
    TaskType.classification: (EvalMetric.auroc, EvalMetric.auprc),
    TaskType.regression: (
        EvalMetric.mae,
        EvalMetric.rmse,
        EvalMetric.spearman,
        EvalMetric.r2,
    ),
}


def metrics_with_headline(
    headline: EvalMetric, task_type: TaskType
) -> list[EvalMetric]:
    """``headline`` first, then the rest of its task type's family, without repeats."""
    ordered = [headline]
    for metric in METRICS_BY_TASK_TYPE.get(task_type, ()):
        if metric not in ordered:
            ordered.append(metric)
    return ordered


class IdentityLabelTransform(BaseModel):
    """The default: the label is scored on the scale it is stored on."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["identity"] = "identity"

    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)


class Log10LabelTransform(BaseModel):
    """``log10(max(y, clip_minimum) + offset)``; NaN (a missing label) stays NaN.

    The defaults reproduce the ASAP / Polaris antiviral-admet-2025 scorer,
    ``np.log10(np.clip(y, a_min=0, a_max=None) + 1)``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["log10"] = "log10"
    clip_minimum: float = 0.0
    offset: float = 1.0

    @model_validator(mode="after")
    def check_positive_argument(self) -> Log10LabelTransform:
        if not self.clip_minimum + self.offset > 0:
            raise ValueError(
                f"log10 transform needs clip_minimum + offset > 0, got "
                f"clip_minimum={self.clip_minimum} and offset={self.offset}"
            )
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        # np.clip and np.log10 both propagate NaN, so missing labels stay missing.
        clipped = np.clip(np.asarray(values, dtype=float), self.clip_minimum, None)
        return np.log10(clipped + self.offset)


#: How a stored (raw) label is mapped before fitting and scoring.
LabelTransform = Annotated[
    IdentityLabelTransform | Log10LabelTransform, Field(discriminator="kind")
]


class LabelColumn(BaseModel):
    """One label column of ``table.parquet``.

    ``n_classes`` is required for, and only meaningful on, a ``multiclass``
    label; the column then stores the integer class index as a ``float64`` and
    invariant 7 checks it lies in ``0 … n_classes - 1``. A ``classification``
    label holds only 0 and 1. ``class_names`` is the optional display labelling
    of the classes, in class-index order; presentation only.

    ``transform`` maps the stored (raw) label to the scale it is fitted and
    scored on; the table always holds raw values. The identity default is left
    out of the serialised form, so a plain label dumps as it always has.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    task_type: TaskType
    n_classes: int | None = None
    class_names: list[str] | None = None
    transform: LabelTransform = Field(default_factory=IdentityLabelTransform)

    @model_serializer(mode="wrap")
    def omit_identity_transform(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        dumped = handler(self)
        if isinstance(self.transform, IdentityLabelTransform):
            dumped.pop("transform", None)
        return dumped

    @model_validator(mode="after")
    def check_transform_task_type(self) -> LabelColumn:
        if (
            not isinstance(self.transform, IdentityLabelTransform)
            and self.task_type is not TaskType.regression
        ):
            raise ValueError(
                f"label {self.name!r} is {self.task_type.value}; a "
                f"{self.transform.kind} transform is only allowed on regression labels"
            )
        return self

    @model_validator(mode="after")
    def check_class_count(self) -> LabelColumn:
        if self.task_type is TaskType.multiclass:
            if self.n_classes is None:
                raise ValueError(
                    f"label {self.name!r} is multiclass and must declare n_classes"
                )
            if self.n_classes < 2:
                raise ValueError(
                    f"label {self.name!r} declares n_classes={self.n_classes}, "
                    "which must be at least 2"
                )
            if self.class_names is not None and len(self.class_names) != self.n_classes:
                raise ValueError(
                    f"label {self.name!r} declares n_classes={self.n_classes} but "
                    f"{len(self.class_names)} class_names; give one name per class "
                    "or none at all"
                )
        else:
            if self.n_classes is not None:
                raise ValueError(
                    f"label {self.name!r} is {self.task_type.value} and must not "
                    "declare n_classes"
                )
            if self.class_names is not None:
                raise ValueError(
                    f"label {self.name!r} is {self.task_type.value} and must not "
                    "declare class_names"
                )
        return self


class EvaluationSpec(BaseModel):
    """How a dataset is scored. Absent on a corpus that is only trained on."""

    model_config = ConfigDict(extra="forbid")

    #: Every metric to compute; the first is the headline cell.
    metrics: list[EvalMetric] = Field(min_length=1)
    split_columns: list[str] = Field(min_length=1)
    default_split: str
    # Declared per dataset with no default: molecule_id for the chiral
    # benchmarks, stereoisomer_id for the drug-property panel, which keeps the
    # public row-level splits (§1.1).
    split_group: SplitGroup
    #: Orphan rule at conformer generation: drop a stereoisomer whose mirror
    #: partner failed, instead of only nulling its ``enantiomer_of``.
    require_enantiomer_pairs: bool = False

    @model_validator(mode="after")
    def check_split_columns(self) -> EvaluationSpec:
        if self.default_split not in self.split_columns:
            raise ValueError(
                f"default_split {self.default_split!r} is not one of "
                f"split_columns {self.split_columns}"
            )
        for column_name in self.split_columns:
            if column_name != "split" and not (
                column_name.startswith("split__") and len(column_name) > len("split__")
            ):
                raise ValueError(
                    f"split column {column_name!r} must be named 'split' or "
                    "'split__<variant>'"
                )
        return self


class DatasetSpec(BaseModel):
    """``dataset.yaml``. A reader refuses any ``format_version`` but 1."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    description: str = ""
    #: Opt-in: the table carries ``isomeric_smiles`` and ``nonisomeric_smiles``.
    smiles: bool = False
    #: Set on a dataset (structures exist), unset on a bundle.
    geometry_origin: GeometryOrigin | None = None
    labels: list[LabelColumn] = Field(default_factory=list)
    extra_columns: list[str] = Field(default_factory=list)
    evaluation: EvaluationSpec | None = None
    # Provenance only. The reader never branches on this.
    source_kind: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_consistency(self) -> DatasetSpec:
        names = self.label_names()
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate label names in {names}")
        if self.evaluation is not None and not self.labels:
            raise ValueError("an evaluation block needs at least one label to score")
        seen: set[str] = set()
        for column_name in self.expected_columns():
            if column_name in seen:
                raise ValueError(
                    f"column name {column_name!r} is declared more than once across "
                    "the fixed columns, labels, split columns and extra columns"
                )
            seen.add(column_name)
        return self

    @property
    def has_structures(self) -> bool:
        """True for a dataset (one row per structure), False for a bundle."""
        return self.geometry_origin is not None

    def split_columns(self) -> list[str]:
        return [] if self.evaluation is None else list(self.evaluation.split_columns)

    def fixed_columns(self) -> list[str]:
        """The leading columns: ids, SMILES if declared, charge if structures exist."""
        columns = ["structure_id"] if self.has_structures else []
        columns += IDENTITY_COLUMNS
        if self.smiles:
            columns += SMILES_COLUMNS
        if self.has_structures:
            columns += CHARGE_COLUMNS
        return columns

    def expected_columns(self) -> list[str]:
        """The exact ordered column list ``table.parquet`` must have (invariant 7)."""
        return [
            *self.fixed_columns(),
            *self.label_names(),
            *self.split_columns(),
            *self.extra_columns,
        ]

    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    def with_structures(self, geometry_origin: GeometryOrigin) -> DatasetSpec:
        """The spec of the dataset built from this bundle."""
        return DatasetSpec.model_validate(
            {**self.model_dump(), "geometry_origin": geometry_origin}
        )

    def task_set(self) -> TaskSet:
        """The zarr-side :class:`TaskSet` the labels become: all per structure."""
        return TaskSet.from_list(
            [
                TaskConfig(
                    name=label.name, task_type=label.task_type, scope=TaskScope.system
                )
                for label in self.labels
            ]
        )
