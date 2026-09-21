"""``benchmark.yaml`` — the :class:`BenchmarkSpec` of a prepared benchmark bundle.

See ``BENCHMARK_DATA_FORMAT.md`` §1.3. The spec says *what a bundle is and how to
score it*; everything source-specific (csv names, column names, downloader
arguments) lives in the preparer and never reaches this package.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from remedi.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)

#: The six fixed leading columns of ``table.parquet`` (§1.1), in order.
FIXED_COLUMNS: tuple[str, ...] = (
    "structure_id",
    "stereoisomer_id",
    "molecule_id",
    "isomeric_smiles",
    "nonisomeric_smiles",
    "enantiomer_of",
)

#: The only values a split column may hold (§1.1 invariant 5).
SPLIT_VALUES: frozenset[str] = frozenset({"train", "valid", "test", "unassigned"})

#: The bundle stage names (§1.2).
BundleStage = Literal["smiles", "conformers"]

#: Where the coordinates of a ``conformers``-stage bundle came from (§1.3).
GeometryOrigin = Literal["source", "etkdg_mmff"]

#: The identity level a split column must be constant within (§1.1 invariant 6).
SplitGroup = Literal["molecule_id", "stereoisomer_id"]


class EvalMetric(StrEnum):
    """Metrics a bundle may ask for. Values are the strings written to yaml.

    Carried over verbatim from the deleted ``remedi.data_handling.benchmarks``
    so that the registry's removal changed no recorded metric name.
    """

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


class BenchmarkTask(BaseModel):
    """One scored column of ``table.parquet``.

    ``n_classes`` is required for, and only meaningful on, a ``multiclass``
    task; the column then stores the integer class index as a ``float64``
    (§1.1) and invariant 7 checks it lies in ``0 … n_classes - 1``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    task_type: TaskType
    n_classes: int | None = None

    @model_validator(mode="after")
    def check_class_count(self) -> BenchmarkTask:
        if self.task_type is TaskType.multiclass:
            if self.n_classes is None:
                raise ValueError(
                    f"task {self.name!r} is multiclass and must declare n_classes"
                )
            if self.n_classes < 2:
                raise ValueError(
                    f"task {self.name!r} declares n_classes={self.n_classes}, "
                    "which must be at least 2"
                )
        elif self.n_classes is not None:
            raise ValueError(
                f"task {self.name!r} is {self.task_type.value} and must not "
                "declare n_classes"
            )
        return self


class BenchmarkSpec(BaseModel):
    """``benchmark.yaml`` (§1.3). A reader refuses any ``format_version`` but 1."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    description: str = ""

    tasks: list[BenchmarkTask] = Field(min_length=1)
    metrics: list[EvalMetric] = Field(min_length=1)

    stage: BundleStage
    geometry_origin: GeometryOrigin | None = None

    split_columns: list[str] = Field(min_length=1)
    default_split: str
    # Declared per bundle with no default: molecule_id for the chiral benchmarks,
    # stereoisomer_id for the drug-property panel, which keeps the public
    # row-level splits (§1.1).
    split_group: SplitGroup
    require_enantiomer_pairs: bool = False

    extra_columns: list[str] = Field(default_factory=list)

    # Provenance only. The reader never branches on this.
    source_kind: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_consistency(self) -> BenchmarkSpec:
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
        if self.stage == "conformers" and self.geometry_origin is None:
            raise ValueError("a conformers-stage bundle must declare geometry_origin")
        if self.stage == "smiles" and self.geometry_origin is not None:
            raise ValueError("a smiles-stage bundle must leave geometry_origin unset")

        task_names = [task.name for task in self.tasks]
        if len(set(task_names)) != len(task_names):
            raise ValueError(f"duplicate task names in {task_names}")

        seen: set[str] = set()
        for column_name in self.expected_columns():
            if column_name in seen:
                raise ValueError(
                    f"column name {column_name!r} is declared more than once across "
                    "the fixed columns, tasks, split columns and extra columns"
                )
            seen.add(column_name)
        return self

    def expected_columns(self) -> list[str]:
        """The exact ordered column list ``table.parquet`` must have (invariant 7)."""
        return [
            *FIXED_COLUMNS,
            *(task.name for task in self.tasks),
            *self.split_columns,
            *self.extra_columns,
        ]

    def task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def task_set(self) -> TaskSet:
        """The zarr-side :class:`TaskSet` these tasks become at ingest.

        Every bundle task is a per-structure label, so all of them land in
        ``system_cols``.
        """
        return TaskSet.from_list(
            [
                TaskConfig(
                    name=task.name, task_type=task.task_type, scope=TaskScope.system
                )
                for task in self.tasks
            ]
        )
