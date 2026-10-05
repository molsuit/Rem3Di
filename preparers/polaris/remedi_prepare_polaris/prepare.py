"""Polaris hub parquet dumps -> one bundle per dataset.

``BENCHMARK_DATA_FORMAT.md`` §5c and §11. Step two of two: ``dump_polaris.py``
(next to this package, in its own polaris-only env) has turned each hub dataset
into ``<dataset_id>.parquet`` plus ``<dataset_id>.source.yaml`` under the raw
root. For each bundle listed in ``datasets.yaml`` next to this module, this
reads the parquet, drops the rows whose CXSMILES carries an OR stereo group (a
separated enantiomer of unknown absolute configuration), makes every AND-group
centre unspecified (§11.3), merges the source rows into one row per stereoisomer
(:func:`remedi.data_handling.bundle.merge_source_rows`), fixes the test fold
(the shipped ``Set`` column or a DeepChem scaffold split), freezes five seeded
train/valid partitions as ``split__seed{1..5}`` and writes
``benchmark_data/bundles/<dataset_id>/``.

Run from the repository root::

    uv run --no-sync prepare-polaris [--only polaris_adme_fang ...]

This never calls polaris: the dumps must already be in place.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rdkit import rdBase

from remedi.data_handling.bundle import (
    DatasetSpec,
    EvalMetric,
    FileHash,
    LabelColumn,
    MergedRows,
    PreparerRecord,
    SourceRecord,
    merge_source_rows,
    sha256_of_file,
)
from remedi.data_handling.bundle.preparation import (
    PREPARER_REPOSITORY,
    BundleReport,
    PreparationError,
    PreparerSettings,
    SourceLabel,
    check_unique,
    preparer_main,
    seeded_split_notice,
    smiles_bundle_spec,
    smiles_evaluation_metrics,
    write_smiles_bundle,
)
from remedi.data_handling.chemistry.splits import (
    fixed_test_seeded_split_columns,
    scaffold_test_mask,
)
from remedi.data_handling.chemistry.stereo_groups import (
    has_or_stereo_group,
    unspecify_relative_stereo,
)
from remedi.data_handling.dataset.tasks import Split, TaskType

logger = logging.getLogger("preparers.polaris")

#: The ``Rem3Di`` checkout this module lives in; the default data paths hang off it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "polaris"
DEFAULT_BUNDLE_ROOT = REPOSITORY_ROOT / "benchmark_data" / "bundles"
DATASETS_FILE = Path(__file__).with_name("datasets.yaml")
SOURCE_KIND = "polaris_hub"
SOURCE_RECORD_SUFFIX = ".source.yaml"
#: ``counts.dropped`` key for a row whose CXSMILES has an OR stereo group: one
#: enantiomer of a chiral separation whose absolute configuration nobody
#: assigned. Unspecifying it would merge the two enantiomers' labels (§11.3).
OR_STEREO_GROUP = "or_stereo_group"

#: The two fixed columns ``dump_polaris.py`` writes before the task columns;
#: the split column holds :class:`Split` codes.
SMILES_COLUMN = "smiles"
SPLIT_CODE_COLUMN = "split"


class PolarisPreparationError(PreparationError):
    """Raised when a dataset cannot be turned into a valid bundle."""


# ------------------------------------------------------------- the catalog


class ShippedTestFold(BaseModel):
    """The test fold is the dataset's own ``Set == test`` rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["shipped"] = "shipped"


class DeepchemScaffoldTestFold(BaseModel):
    """The test fold is the test part of the DeepChem scaffold split.

    Computed over the final bundle rows (after the filter and the merge, in
    first-occurrence order), so it is a pure function of the bundle's SMILES.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["deepchem_scaffold"] = "deepchem_scaffold"
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    valid_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def check_fractions(self) -> DeepchemScaffoldTestFold:
        if self.train_fraction + self.valid_fraction >= 1.0:
            raise ValueError("train_fraction + valid_fraction leaves no test fold")
        return self


TestFold = Annotated[
    ShippedTestFold | DeepchemScaffoldTestFold, Field(discriminator="kind")
]


class PolarisDataset(BaseModel):
    """One bundle: which parquet, which labels, how it is split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    #: The parquet file name under the raw root; several bundles may share one.
    source_file: str = Field(min_length=1)
    slug: str = Field(min_length=1)
    property_description: str = Field(min_length=1)
    task_type: Literal[TaskType.regression, TaskType.classification]
    test_fold: TestFold
    labels: list[SourceLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def check_labels(self) -> PolarisDataset:
        check_unique([label.name for label in self.labels], "label names")
        check_unique([label.column for label in self.labels], "label source columns")
        # LabelColumn owns the transform / task type rule; build them once here
        # so a bad catalog fails at load time rather than mid-run.
        self.label_columns()
        return self

    @property
    def source_record_file(self) -> str:
        """``<stem>.source.yaml``: what ``dump_polaris.py`` wrote next to the parquet."""
        return self.source_file.removesuffix(".parquet") + SOURCE_RECORD_SUFFIX

    def label_columns(self) -> list[LabelColumn]:
        return [label.label_column(self.task_type) for label in self.labels]

    def metrics(self) -> list[EvalMetric]:
        return smiles_evaluation_metrics(
            self.task_type,
            len(self.labels),
            EvalMetric.mae
            if self.task_type is TaskType.regression
            else EvalMetric.auroc,
        )


class PolarisDatasetCatalog(BaseModel):
    """``datasets.yaml``: every bundle this preparer knows, in run order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[PolarisDataset] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_dataset_ids(self) -> PolarisDatasetCatalog:
        check_unique(
            [dataset.dataset_id for dataset in self.datasets], "dataset_id values"
        )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> PolarisDatasetCatalog:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


#: The five Polaris bundles (§5c).
POLARIS_DATASETS: tuple[PolarisDataset, ...] = tuple(
    PolarisDatasetCatalog.from_yaml(DATASETS_FILE).datasets
)


class PolarisPreparerConfig(PreparerSettings):
    """Knobs of one preparer run."""

    #: Directory holding the ``dump_polaris.py`` parquets and source yamls.
    raw_root: Path = DEFAULT_RAW_ROOT
    bundle_root: Path = DEFAULT_BUNDLE_ROOT
    #: The valid share of the non-test rows in every seeded partition; 1/9 of
    #: the 90 % left by a 10 % test fold is 10 % of the whole.
    valid_fraction: float = Field(default=1 / 9, gt=0.0, lt=1.0)

    def dataset_ids(self) -> list[str]:
        return [dataset.dataset_id for dataset in POLARIS_DATASETS]

    def selected_datasets(self) -> list[PolarisDataset]:
        wanted = set(self.selected_dataset_ids())
        return [dataset for dataset in POLARIS_DATASETS if dataset.dataset_id in wanted]


# ------------------------------------------------------------ source tables


class PolarisSourceRecord(BaseModel):
    """``<dataset_id>.source.yaml`` as ``dump_polaris.py`` writes it."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    polaris_lib_version: str
    #: The hub's checksum of the dataset; ``None`` when it publishes none.
    checksum: str | None = None
    dumped_at: datetime


@dataclass(frozen=True)
class SourceTable:
    """One dumped parquet and the record written next to it."""

    frame: pd.DataFrame
    record: PolarisSourceRecord
    file_hashes: dict[str, FileHash]

    def __len__(self) -> int:
        return len(self.frame)


def read_source_table(raw_root: Path, dataset: PolarisDataset) -> SourceTable:
    """Read one dumped parquet and check it against the catalog.

    Raises:
        PolarisPreparationError: on a missing file, a malformed or foreign
            source record, an unknown split code, or a missing or non-numeric
            label column.
    """
    parquet_path = raw_root / dataset.source_file
    record_path = raw_root / dataset.source_record_file
    for path in (parquet_path, record_path):
        if not path.is_file():
            raise PolarisPreparationError(
                f"missing raw file {path}; run preparers/polaris/dump_polaris.py "
                "first (this preparer never calls polaris)"
            )
    try:
        record = PolarisSourceRecord.model_validate(
            yaml.safe_load(record_path.read_text())
        )
    except ValidationError as error:
        raise PolarisPreparationError(f"{record_path} is malformed: {error}") from error
    if record.slug != dataset.slug:
        raise PolarisPreparationError(
            f"{dataset.dataset_id}: {record_path.name} was dumped from "
            f"{record.slug!r}, the catalog expects {dataset.slug!r}"
        )

    frame = pd.read_parquet(parquet_path)
    unknown_codes = sorted(
        set(frame[SPLIT_CODE_COLUMN].astype(int)) - {split.value for split in Split}
    )
    if unknown_codes:
        raise PolarisPreparationError(
            f"{parquet_path.name} has unknown split codes {unknown_codes}"
        )
    for label in dataset.labels:
        if label.column not in frame.columns or not pd.api.types.is_numeric_dtype(
            frame[label.column]
        ):
            raise PolarisPreparationError(
                f"{dataset.dataset_id}: label column {label.column!r} is missing "
                f"from {parquet_path.name} or not numeric (columns: {list(frame.columns)})"
            )
    return SourceTable(
        frame=frame,
        record=record,
        file_hashes={
            path.name: FileHash(sha256=sha256_of_file(path))
            for path in (parquet_path, record_path)
        },
    )


# ---------------------------------------------------------- enhanced stereo


@dataclass(frozen=True)
class UnspecifiedStereoRows:
    """Every raw SMILES with its AND-group (racemate) stereo made unspecified."""

    smiles: list[str]
    #: Source rows that lost at least one stereo assignment.
    rows_changed: int
    #: Stereocentres and stereo bonds made unspecified, over all rows.
    centres_unspecified: int

    def notice(self) -> str:
        return (
            "CXSMILES enhanced stereo: rows with an OR (o) group, single enantiomers "
            "of unknown absolute configuration, are dropped "
            f"(counts.dropped.{OR_STEREO_GROUP}); every stereocentre in an AND (&) "
            "group, a racemate, is made unspecified before the SMILES filter, and "
            f"absolute (a) centres keep their configuration. {self.rows_changed} "
            f"kept source rows had {self.centres_unspecified} centres unspecified in "
            "total; rows that then canonicalise identically are merged like any "
            "replicate."
        )


def unspecify_stereo_groups(raw_smiles: Sequence[str]) -> UnspecifiedStereoRows:
    """Run :func:`unspecify_relative_stereo` over every raw SMILES.

    An unparseable SMILES passes on unchanged so the SMILES filter counts it as
    invalid.
    """
    smiles: list[str] = []
    rows_changed = centres_unspecified = 0
    for raw in raw_smiles:
        outcome = unspecify_relative_stereo(raw)
        if outcome is None:
            smiles.append(raw)
            continue
        smiles.append(outcome.smiles)
        if outcome.unspecified_centres:
            rows_changed += 1
            centres_unspecified += outcome.unspecified_centres
    return UnspecifiedStereoRows(smiles, rows_changed, centres_unspecified)


# ------------------------------------------------------------- the test fold


@dataclass(frozen=True)
class TestFoldAssignment:
    """Which bundle rows are in the fixed test fold, and how it was decided."""

    is_test: np.ndarray
    #: How the fold was made, for the provenance notice.
    description: str
    #: Bundle rows merging shipped-test and other source rows; ``None`` when derived.
    straddling_rows: int | None = None


def assign_test_fold(
    dataset: PolarisDataset, merged: MergedRows, split_codes: np.ndarray
) -> TestFoldAssignment:
    """The fixed test fold the catalog declares, checked against the parquet.

    A shipped split makes a bundle row test if *any* source row merged into it
    was shipped test; shipped valid rows join the pool the seeded partitions
    re-split.

    Raises:
        PolarisPreparationError: if the catalog's kind disagrees with the data
            (split codes present or not), or a shipped split leaves rows
            unassigned.
    """
    unassigned = int((split_codes == Split.unassigned.value).sum())
    ships_split = unassigned < len(split_codes)
    match dataset.test_fold:
        case ShippedTestFold():
            if not ships_split:
                raise PolarisPreparationError(
                    f"{dataset.dataset_id}: the catalog declares a shipped test fold "
                    "but the parquet ships no split; declare 'deepchem_scaffold'"
                )
            if unassigned:
                raise PolarisPreparationError(
                    f"{dataset.dataset_id}: {unassigned} of {len(split_codes)} source "
                    "rows have no shipped split; every row must be train, valid or test"
                )
            member_is_test = [
                [split_codes[row] == Split.test.value for row in members]
                for members in merged.member_source_rows
            ]
            straddling = sum(
                1 for flags in member_is_test if any(flags) and not all(flags)
            )
            return TestFoldAssignment(
                is_test=np.array([any(flags) for flags in member_is_test], dtype=bool),
                description=(
                    "the dataset's shipped Set column, a bundle row being test if any "
                    f"source row merged into it was; {straddling} bundle rows merge "
                    "shipped test and other source rows"
                ),
                straddling_rows=straddling,
            )
        case DeepchemScaffoldTestFold() as scaffold:
            if ships_split:
                raise PolarisPreparationError(
                    f"{dataset.dataset_id}: the parquet ships split codes but the "
                    "catalog declares a deepchem_scaffold test fold; declare 'shipped'"
                )
            return TestFoldAssignment(
                is_test=scaffold_test_mask(
                    merged.isomeric_smiles,
                    scaffold.train_fraction,
                    scaffold.valid_fraction,
                ),
                description=(
                    "the dataset ships no split, so the test part of the deterministic "
                    f"DeepChem scaffold split ({scaffold.train_fraction:g}/"
                    f"{scaffold.valid_fraction:g}, chirality-aware scaffolds) over the "
                    "final bundle rows"
                ),
            )


# -------------------------------------------------------------- the dataset


def build_spec(dataset: PolarisDataset, split_columns: list[str]) -> DatasetSpec:
    """``dataset.yaml`` for one bundle."""
    test_fold_text = (
        "the fixed shipped test fold"
        if isinstance(dataset.test_fold, ShippedTestFold)
        else "a fixed DeepChem scaffold test fold (the source ships no split)"
    )
    return smiles_bundle_spec(
        dataset_id=dataset.dataset_id,
        description=(
            f"{dataset.property_description}. From the Polaris hub dataset "
            f"{dataset.slug}. Rows are the source rows after the Rem3Di SMILES "
            "filter (CXSMILES OR-group rows dropped, AND-group centres made "
            "unspecified first), one row "
            "per stereoisomer with replicate measurements aggregated and missing "
            f"labels left empty; {test_fold_text} plus five seeded scaffold "
            "train/valid partitions are frozen as the split columns."
        ),
        labels=dataset.label_columns(),
        metrics=dataset.metrics(),
        split_columns=split_columns,
        source_kind=SOURCE_KIND,
    )


def prepare_dataset(
    dataset: PolarisDataset, config: PolarisPreparerConfig, preparer: PreparerRecord
) -> BundleReport:
    """Read, drop OR-group rows, unspecify, merge, split and write one bundle.

    Raises:
        PolarisPreparationError: on a source problem this preparer detects.
        LabelAggregationError: on a classification label other than 0 or 1.
        BundleValidationError: if the table violates a format invariant.
    """
    source = read_source_table(config.raw_root, dataset)
    is_or_row = source.frame[SMILES_COLUMN].astype(str).map(has_or_stereo_group)
    frame = source.frame[~is_or_row].reset_index(drop=True)
    stereo = unspecify_stereo_groups(frame[SMILES_COLUMN].astype(str).tolist())
    merged = merge_source_rows(
        stereo.smiles,
        {
            label.name: frame[label.column].astype(float).tolist()
            for label in dataset.labels
        },
        {label.name: dataset.task_type for label in dataset.labels},
        config.smiles_filter,
    ).after_source_drops({OR_STEREO_GROUP: int(is_or_row.sum())})
    test_fold = assign_test_fold(
        dataset, merged, frame[SPLIT_CODE_COLUMN].to_numpy(dtype=np.int64)
    )
    return write_smiles_bundle(
        spec=build_spec(dataset, config.split_columns()),
        merged=merged,
        split_values=fixed_test_seeded_split_columns(
            merged.isomeric_smiles,
            test_fold.is_test,
            list(config.seeds),
            config.valid_fraction,
        ),
        settings=config,
        preparer=preparer,
        source=SourceRecord(
            files=source.file_hashes,
            package_versions={
                "polaris-lib": source.record.polaris_lib_version,
                "rdkit": rdBase.rdkitVersion,
            },
        ),
        notices=[
            stereo.notice(),
            seeded_split_notice(
                test_fold.description, config.seeds, config.valid_fraction
            ),
        ],
        extra={
            "stereo_unspecified_rows": stereo.rows_changed,
            "test_straddling_rows": (
                "-" if test_fold.straddling_rows is None else test_fold.straddling_rows
            ),
        },
    )


def run(config: PolarisPreparerConfig) -> list[BundleReport]:
    """Prepare every selected dataset. Returns one report each."""
    preparer = PreparerRecord.for_script(PREPARER_REPOSITORY, Path(__file__))
    reports: list[BundleReport] = []
    for dataset in config.selected_datasets():
        logger.info("preparing %s", dataset.dataset_id)
        reports.append(prepare_dataset(dataset, config, preparer))
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    return preparer_main(
        argv, description=__doc__ or "", settings_type=PolarisPreparerConfig, run=run
    )


if __name__ == "__main__":
    sys.exit(main())
