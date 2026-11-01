"""What every SMILES preparer shares: settings, the table, the spec, the run report.

A SMILES preparer (TDC, MoleculeNet, Polaris) reads its source, hands the rows to
:func:`~remedi.data_handling.bundle.aggregation.merge_source_rows`, makes its
split columns, and writes a bundle. Everything except reading the source and
making the splits is the same for all of them and lives here, so that a
preparer module reads as exactly those source-specific steps.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from remedi.data_handling.bundle.aggregation import (
    AGGREGATION_NOTICE,
    MEASUREMENT_COUNT_COLUMN,
    MergedRows,
)
from remedi.data_handling.bundle.bundle import (
    Bundle,
    BundleValidationError,
    write_bundle,
)
from remedi.data_handling.bundle.provenance import (
    BundleCounts,
    BundleProvenance,
    PreparerRecord,
    SourceRecord,
)
from remedi.data_handling.bundle.spec import (
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    IdentityLabelTransform,
    LabelColumn,
    LabelTransform,
    metrics_with_headline,
)
from remedi.data_handling.chemistry.smiles_filter import SmilesFilterConfig
from remedi.data_handling.dataset.tasks import TaskType

logger = logging.getLogger(__name__)

#: Recorded as ``preparer.repo`` in every bundle's provenance.
PREPARER_REPOSITORY = "molsuit/Rem3Di"


class PreparationError(RuntimeError):
    """Raised when a source cannot be turned into a valid bundle."""


def check_unique(values: Sequence[str], what: str) -> None:
    """Raise ``ValueError`` naming the duplicates if ``values`` repeat (catalog checks)."""
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what} {duplicates}")


# ---------------------------------------------------------------- settings


class SourceLabel(BaseModel):
    """One label column of a bundle and the source column it is read from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The bundle label column.
    name: str = Field(min_length=1)
    #: The source column, when it differs from ``name``.
    source_column: str | None = None
    transform: LabelTransform = Field(default_factory=IdentityLabelTransform)

    @property
    def column(self) -> str:
        return self.name if self.source_column is None else self.source_column

    def label_column(self, task_type: TaskType) -> LabelColumn:
        return LabelColumn(
            name=self.name, task_type=task_type, transform=self.transform
        )


class PreparerSettings(BaseModel):
    """The knobs every SMILES preparer run has; each preparer subclasses it.

    A subclass gives ``raw_root`` and ``bundle_root`` their defaults and
    implements :meth:`dataset_ids` (its catalog, in run order).
    """

    model_config = ConfigDict(extra="forbid")

    raw_root: Path
    #: Directory that receives one ``<dataset_id>/`` bundle per dataset.
    bundle_root: Path
    #: The seeded train/valid partitions frozen into the bundle; the bare
    #: ``split`` column aliases the first.
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5], min_length=1)
    #: Recorded verbatim in ``provenance.yaml``. ``dedupe`` is off because the
    #: merge collapses duplicates itself (``filter_source_smiles``).
    smiles_filter: SmilesFilterConfig = Field(
        default_factory=lambda: SmilesFilterConfig(dedupe=False)
    )
    #: Restrict the run to these dataset ids; empty means the whole catalog.
    only: list[str] = Field(default_factory=list)

    def dataset_ids(self) -> list[str]:
        raise NotImplementedError

    @model_validator(mode="after")
    def check_seeds_and_selection(self) -> PreparerSettings:
        check_unique([str(seed) for seed in self.seeds], "seeds")
        known = self.dataset_ids()
        unknown = sorted(set(self.only) - set(known))
        if unknown:
            raise ValueError(f"unknown dataset ids {unknown}; known: {sorted(known)}")
        return self

    def selected_dataset_ids(self) -> list[str]:
        """The datasets this run prepares, in catalog order."""
        wanted = set(self.only)
        return [
            dataset_id
            for dataset_id in self.dataset_ids()
            if not wanted or dataset_id in wanted
        ]

    def split_columns(self) -> list[str]:
        """``[split, split__seed1, …]``: the bare name first, then one per seed."""
        return ["split", *(f"split__seed{seed}" for seed in self.seeds)]


def seeded_split_notice(
    test_fold: str, seeds: Sequence[int], valid_fraction: float
) -> str:
    """The provenance notice for a fixed test fold plus seeded train/valid partitions."""
    return (
        f"Test fold: {test_fold}; it is the same in every split column. The other "
        "rows are partitioned into train and valid once per seed "
        f"({', '.join(str(seed) for seed in seeds)}) by a seeded balanced scaffold "
        f"split with a valid share of {valid_fraction:.4g} of the non-test rows; "
        f"the bare split column aliases split__seed{seeds[0]}."
    )


# ---------------------------------------------------------- table and spec


def smiles_evaluation_metrics(
    task_type: TaskType, label_count: int, headline: EvalMetric
) -> list[EvalMetric]:
    """``headline`` and its family, or macro-AUROC alone for multi-label classification.

    The eval scores a multi-label classification dataset on its 2-D label
    matrix at once, and only macro-AUROC handles the missing (NaN) cells there.
    Multi-target regression is scored per column, so it keeps the family.
    """
    if task_type is TaskType.classification and label_count > 1:
        return [EvalMetric.macro_auroc]
    return metrics_with_headline(headline, task_type)


def smiles_bundle_spec(
    dataset_id: str,
    description: str,
    labels: list[LabelColumn],
    metrics: list[EvalMetric],
    split_columns: list[str],
    source_kind: str,
) -> DatasetSpec:
    """``dataset.yaml`` of a SMILES bundle: one row per stereoisomer, public splits."""
    return DatasetSpec(
        dataset_id=dataset_id,
        description=description,
        smiles=True,
        labels=labels,
        extra_columns=[MEASUREMENT_COUNT_COLUMN],
        evaluation=EvaluationSpec(
            metrics=metrics,
            split_columns=split_columns,
            default_split="split",
            split_group="stereoisomer_id",
            require_enantiomer_pairs=False,
        ),
        source_kind=source_kind,
    )


def smiles_bundle_table(
    merged: MergedRows, split_values: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Identity and SMILES columns, the labels, the split columns, ``n_measurements``.

    Missing labels (NaN) are allowed, a label column without any value is not.

    Raises:
        PreparationError: on an all-missing label column or an ``unassigned``
            split value, both silent corruption downstream.
    """
    table = merged.aggregated.identity.to_frame()
    for name, values in merged.aggregated.labels.items():
        table[name] = pd.Series(list(values), dtype="float64", index=table.index)
        if table[name].isna().all():
            raise PreparationError(f"label column {name!r} has no value at all")
    for column_name, values in split_values.items():
        table[column_name] = pd.Series(list(values), dtype="str", index=table.index)
        if (table[column_name] == "unassigned").any():
            raise PreparationError(f"split column {column_name!r} has unassigned rows")
    table[MEASUREMENT_COUNT_COLUMN] = pd.Series(
        merged.aggregated.measurement_counts, dtype="float64", index=table.index
    )
    return table


# ------------------------------------------------------------ write + report


@dataclass(frozen=True)
class BundleReport:
    """What one written bundle came out as, for the run summary."""

    spec: DatasetSpec
    #: The provenance as written, counts and output hashes filled in.
    provenance: BundleProvenance
    directory: Path
    #: How many bundle rows carry more than one measurement.
    rows_with_replicates: int
    #: Source-specific report columns (header -> value).
    extra: dict[str, int | str] = field(default_factory=dict)

    @property
    def counts(self) -> BundleCounts:
        return self.provenance.counts

    @property
    def content_sha256(self) -> str:
        if self.provenance.outputs is None:  # pragma: no cover - the writer fills it
            raise PreparationError(f"{self.directory} has no output hashes")
        return self.provenance.outputs.table_parquet.content_sha256


def write_smiles_bundle(
    spec: DatasetSpec,
    merged: MergedRows,
    split_values: Mapping[str, Sequence[str]],
    settings: PreparerSettings,
    preparer: PreparerRecord,
    source: SourceRecord,
    notices: Sequence[str] = (),
    extra: Mapping[str, int | str] | None = None,
) -> BundleReport:
    """Build the table and provenance and write the bundle.

    Raises:
        PreparationError: if the table is malformed (:func:`smiles_bundle_table`).
        BundleValidationError: if it violates a format invariant.
    """
    provenance = BundleProvenance(
        dataset_id=spec.dataset_id,
        preparer=preparer,
        source=source,
        smiles_filter=settings.smiles_filter,
        counts=BundleCounts(
            source_molecules=merged.source_rows, dropped=merged.dropped
        ),
        notices=[AGGREGATION_NOTICE, *notices],
    )
    directory = settings.bundle_root / spec.dataset_id
    bundle = Bundle(
        spec=spec,
        table=smiles_bundle_table(merged, split_values),
        provenance=provenance,
    )
    return BundleReport(
        spec=spec,
        provenance=write_bundle(bundle, directory),
        directory=directory,
        rows_with_replicates=merged.aggregated.rows_with_replicates,
        extra=dict(extra or {}),
    )


def format_report(reports: Iterable[BundleReport]) -> str:
    """A fixed-width summary of a run, one line per bundle.

    Columns: the dataset, its headline metric and label count, source and final
    rows, every drop reason that occurs, rows with replicates, the default
    split's fold sizes, constitutions straddling folds, the source-specific
    extras, and the start of the content hash.
    """
    reports = list(reports)
    drop_reasons = list(
        dict.fromkeys(reason for report in reports for reason in report.counts.dropped)
    )
    extra_names = list(
        dict.fromkeys(name for report in reports for name in report.extra)
    )
    rows: list[dict[str, str]] = []
    for report in reports:
        counts = report.counts
        evaluation = report.spec.evaluation
        rows.append(
            {
                "dataset_id": report.spec.dataset_id,
                "metric": evaluation.metrics[0].value if evaluation else "",
                "labels": str(len(report.spec.labels)),
                "source": str(counts.source_molecules),
                "final": str(counts.final_rows),
                **{
                    reason: str(counts.dropped.get(reason, 0))
                    for reason in drop_reasons
                },
                "replicated": str(report.rows_with_replicates),
                **{
                    fold: str(counts.per_split.get(fold, 0))
                    for fold in ("train", "valid", "test")
                },
                "straddling": str(counts.stereoisomer_straddling_constitutions),
                **{name: str(report.extra.get(name, "")) for name in extra_names},
                "content_sha256": report.content_sha256[:14],
            }
        )
    if not rows:
        return "(no datasets prepared)"
    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    left_aligned = {"dataset_id", "metric", "content_sha256"}

    def line(cells: Mapping[str, str]) -> str:
        return " ".join(
            cells[header].ljust(widths[header])
            if header in left_aligned
            else cells[header].rjust(widths[header])
            for header in headers
        )

    header_line = line({header: header for header in headers})
    return "\n".join([header_line, "-" * len(header_line), *map(line, rows)])


# ---------------------------------------------------------------------- cli


def preparer_main[SettingsT: PreparerSettings](
    argv: Sequence[str] | None,
    *,
    description: str,
    settings_type: type[SettingsT],
    run: Callable[[SettingsT], list[BundleReport]],
    add_arguments: Callable[[argparse.ArgumentParser], object] | None = None,
    extra_settings: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> int:
    """The command line every preparer shares: ``--raw-root``, ``--bundle-root``, ``--only``.

    ``add_arguments`` adds a preparer's own flags and ``extra_settings`` turns
    them into settings fields. Prints the run report; returns the exit code.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=description)
    fields = settings_type.model_fields
    parser.add_argument("--raw-root", type=Path, default=fields["raw_root"].default)
    parser.add_argument(
        "--bundle-root", type=Path, default=fields["bundle_root"].default
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="DATASET_ID",
        help="prepare only this dataset; repeatable",
    )
    if add_arguments is not None:
        add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        settings = settings_type(
            raw_root=arguments.raw_root,
            bundle_root=arguments.bundle_root,
            only=list(arguments.only),
            **(extra_settings(arguments) if extra_settings is not None else {}),
        )
        reports = run(settings)
    # ValueError covers pydantic's ValidationError, LabelAggregationError and a
    # splitter refusing the data (e.g. an empty test fold).
    except (ValueError, PreparationError, BundleValidationError) as error:
        logger.error("%s", error)
        return 1
    print(format_report(reports))
    return 0
