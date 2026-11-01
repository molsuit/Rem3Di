"""The benchmark protocol: what one benchmark dataset offers to an evaluation.

The panel scores through this module, and so can anything outside it (a
fine-tuning run, another group's model):

.. code-block:: python

    benchmark = load_benchmark(dataset_dir)        # spec + zarr + table, validated
    benchmark.task_kind, benchmark.target_columns, benchmark.headline_metric
    partition = benchmark.partition("split__seed3")  # train / valid / test structure ids
    targets = benchmark.targets(partition.train)     # NaN for missing, transforms applied

Structure ids are the rows of the dataset's ``table.parquet`` and of its zarr
(invariant 1), so a ``Partition``'s sorted id arrays index descriptor matrices
in zarr order directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

from remedi_data.bundle import (
    TABLE_FILENAME,
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    read_spec,
    read_table,
)
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.tasks import Split, TaskType, split_codes_from_names

TaskKind = Literal["regression", "binary", "multilabel", "multiclass"]


@dataclass(frozen=True)
class Partition:
    """One train / valid / test partition as sorted structure-id arrays."""

    split_column: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def evaluation_of(spec: DatasetSpec) -> EvaluationSpec:
    """The spec's evaluation block.

    Raises:
        ValueError: if the dataset declares none (a corpus, not a benchmark).
    """
    if spec.evaluation is None:
        raise ValueError(f"{spec.dataset_id} declares no evaluation block")
    return spec.evaluation


def task_kind(dataset: MoleculeDataset) -> TaskKind:
    """Regression, binary, multilabel or multiclass, from the zarr's task columns.

    Raises:
        ValueError: if the dataset has no task set, mixes task types, or has a
            multiclass task with more than one column.
    """
    if dataset.config.tasks is None:
        raise ValueError("Dataset has no TaskSet; cannot infer task kind.")
    columns = dataset.config.tasks.system_cols
    types = {column.task_type for column in columns}
    if types == {TaskType.regression}:
        # Multi-target regression (e.g. polaris_adme_fang) is scored per column
        # downstream — one independent single-target fit per target.
        return "regression"
    if types == {TaskType.classification}:
        return "binary" if len(columns) == 1 else "multilabel"
    if types == {TaskType.multiclass}:
        # Single-label multi-class: exactly one column carrying integer class
        # indices (e.g. chiral_cat's 5 chirality types).
        if len(columns) != 1:
            raise ValueError(
                f"multiclass benchmark must have one column, got {len(columns)}"
            )
        return "multiclass"
    raise ValueError(f"Mixed task types in one benchmark: {types}")


def apply_label_transforms(
    targets: np.ndarray, column_names: list[str], spec: DatasetSpec
) -> np.ndarray:
    """Each target column mapped through its spec label's ``transform``.

    ``targets`` is ``(n_structures, n_columns)`` with columns named by
    ``column_names``; NaN (missing) stays NaN. Every column must be a label the
    spec declares, so a transform can never be silently skipped.
    """
    if targets.ndim != 2 or targets.shape[1] != len(column_names):
        raise ValueError(
            f"targets of shape {targets.shape} do not match the "
            f"{len(column_names)} target columns {column_names}"
        )
    labels_by_name = {label.name: label for label in spec.labels}
    transformed = np.array(targets, dtype=float, copy=True)
    for column_index, column_name in enumerate(column_names):
        label = labels_by_name.get(column_name)
        if label is None:
            raise ValueError(
                f"target column {column_name!r} is not a label of "
                f"{spec.dataset_id} (labels: {spec.label_names()})"
            )
        transformed[:, column_index] = label.transform.apply(
            transformed[:, column_index]
        )
    return transformed


def build_targets_with_nan(dataset: MoleculeDataset, spec: DatasetSpec) -> np.ndarray:
    """``targets_system`` with NaN where ``mask_system == 0``, label transforms applied.

    Downstream learner code uses NaN to skip missing labels in sparse multi-
    label benchmarks (SIDER / ClinTox / Tox21). The zarr holds raw labels; each
    column goes through its spec label's ``transform`` here, the single place
    targets enter fitting and scoring, so metrics are on the transformed scale.
    """
    if dataset.targets_system is None or dataset.mask_system is None:
        raise ValueError(
            "Dataset has no system task arrays; cannot evaluate. Build with "
            "the benchmark ingest runner so targets are materialized."
        )
    if dataset.config.tasks is None:
        raise ValueError("Dataset has no TaskSet; cannot name its target columns.")
    targets = np.asarray(dataset.targets_system[:], dtype=float)
    mask = np.asarray(dataset.mask_system[:], dtype=bool)
    column_names = [column.name for column in dataset.config.tasks.system_cols]
    return apply_label_transforms(np.where(mask, targets, np.nan), column_names, spec)


def read_partition(
    dataset_dir: Path, dataset: MoleculeDataset, split_column: str
) -> Partition:
    """The train / valid / test structure ids of one split column.

    The split is read by name from the ``table.parquet`` beside the zarr and
    gathered onto the zarr's rows through ``ids/bundle_row``, so any of the
    bundle's split columns can be scored, not just the one materialised into
    ``tasks/split``.

    Raises:
        ValueError: if the dataset carries no table, if ``split_column`` is not
            one of its columns, if its ``structure_id`` column is not the row
            index, or if a split value is not a known split name.
    """
    dataset_dir = Path(dataset_dir)
    n_structures = dataset.N_structures
    try:
        table = read_table(dataset_dir, columns=["structure_id", split_column])
    except FileNotFoundError as error:
        raise ValueError(
            f"{dataset_dir} has no {TABLE_FILENAME}; rebuild it with build_datasets, "
            "which writes the table beside the zarr"
        ) from error
    except Exception as error:  # pyarrow raises its own type for a missing column
        available = [
            name for name in read_table(dataset_dir).columns if name.startswith("split")
        ]
        raise ValueError(
            f"split column {split_column!r} is not in {dataset_dir / TABLE_FILENAME}; "
            f"its split columns are {available}"
        ) from error

    structure_ids = table["structure_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(structure_ids, np.arange(len(table), dtype=np.int64)):
        raise ValueError(
            f"{dataset_dir / TABLE_FILENAME} violates invariant 1: structure_id is not "
            "the row index, so predictions cannot be joined back to it"
        )
    rows = np.asarray(
        dataset.bundle_rows_or_structure_ids[:n_structures], dtype=np.int64
    )
    codes = split_codes_from_names(table[split_column].to_numpy(dtype=object)[rows])
    return Partition(
        split_column=split_column,
        train=np.flatnonzero(codes == Split.train.value),
        valid=np.flatnonzero(codes == Split.valid.value),
        test=np.flatnonzero(codes == Split.test.value),
    )


@dataclass
class Benchmark:
    """One built benchmark dataset: its spec, its zarr and its table."""

    path: Path
    spec: DatasetSpec
    dataset: MoleculeDataset

    @property
    def dataset_id(self) -> str:
        return self.spec.dataset_id

    @property
    def evaluation(self) -> EvaluationSpec:
        return evaluation_of(self.spec)

    @property
    def headline_metric(self) -> EvalMetric:
        return self.evaluation.metrics[0]

    @property
    def default_split(self) -> str:
        return self.evaluation.default_split

    @cached_property
    def task_kind(self) -> TaskKind:
        return task_kind(self.dataset)

    @property
    def target_columns(self) -> list[str]:
        assert self.dataset.config.tasks is not None  # checked by load_benchmark
        return [column.name for column in self.dataset.config.tasks.system_cols]

    @property
    def n_structures(self) -> int:
        return self.dataset.N_structures

    @cached_property
    def _all_targets(self) -> np.ndarray:
        return build_targets_with_nan(self.dataset, self.spec)

    def targets(self, structure_ids: np.ndarray | None = None) -> np.ndarray:
        """``(n, n_target_columns)`` targets, NaN for missing, transforms applied.

        ``None`` returns every structure in zarr order.
        """
        if structure_ids is None:
            return self._all_targets
        return self._all_targets[np.asarray(structure_ids, dtype=np.int64)]

    def partition(self, split_column: str | None = None) -> Partition:
        """The partition of ``split_column``; ``None`` is the spec's ``default_split``."""
        return read_partition(
            self.path, self.dataset, split_column or self.default_split
        )

    def structure_group_ids(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-structure (constitution id, stereoisomer id), in zarr order."""
        return (
            np.asarray(self.dataset.molecule_ids[:]),
            np.asarray(self.dataset.isomer_ids[:]),
        )


def load_benchmark(dataset_dir: Path) -> Benchmark:
    """Open a built benchmark dataset and check what the protocol relies on.

    Raises:
        ValueError: if the dataset declares no evaluation block, has no task
            set, or its zarr ``structure_id`` is not the row index.
    """
    dataset_dir = Path(dataset_dir)
    spec = read_spec(dataset_dir)
    evaluation_of(spec)
    dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
    if dataset.config.tasks is None:
        raise ValueError(f"{dataset_dir} has no TaskSet; it is not a benchmark")
    structure_ids = np.asarray(dataset.structure_ids[:], dtype=np.int64)
    if not np.array_equal(structure_ids, np.arange(len(structure_ids))):
        raise ValueError(
            f"{dataset_dir} violates invariant 1: the zarr's structure_id is not "
            "the row index"
        )
    return Benchmark(path=dataset_dir, spec=spec, dataset=dataset)
