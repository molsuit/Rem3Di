"""Source rows -> one bundle row per stereoisomer, shared by the SMILES preparers.

Every SMILES-bearing preparer (TDC, MoleculeNet, Polaris) does the same three
things between reading its source and writing a bundle: run the shared SMILES
filter over every source row, collapse rows that canonicalise to the same
isomeric SMILES while merging their labels, and check that every source row is
either a bundle row or counted under one drop reason. This module is that
shared step.

The merge rule (``BENCHMARK_DATA_FORMAT.md`` §9 row 17): per label column, the
mean of the replicate measurements for a regression label and their majority
vote for a classification label; missing measurements (NaN) are ignored, and an
exact tie leaves that label missing. A row left with no label at all is dropped
and counted.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from remedi.data_handling.bundle.identity import (
    IdentityTable,
    assign_identity,
    canonical_smiles_pair,
)
from remedi.data_handling.chemistry.smiles_filter import (
    SmilesFilterConfig,
    filter_smiles,
)
from remedi.data_handling.dataset.tasks import TaskType

#: The ``extra_columns`` entry every aggregated bundle carries: how many source
#: rows went into this row's labels (1 for a compound measured once).
MEASUREMENT_COUNT_COLUMN = "n_measurements"

#: ``counts.dropped`` keys.
INVALID_SMILES = "invalid_smiles"
SMILES_FILTER = "smiles_filter"
#: Replicate rows merged into an earlier row.
DUPLICATE_SMILES = "duplicate_smiles"
#: Compounds whose every label was lost to an exact classification tie.
DUPLICATE_LABEL_TIE = "duplicate_label_tie"
#: Compounds the source lists without any label.
MISSING_LABEL = "missing_label"

#: Recorded in ``provenance.yaml`` under ``notices`` on every aggregated bundle.
AGGREGATION_NOTICE = (
    "Replicate rows sharing a canonical isomeric SMILES are merged into one row: "
    "the label is the mean of the replicate measurements for a regression "
    "endpoint and their majority vote for a classification endpoint, a compound "
    "whose classification votes tie exactly is dropped as having no known label "
    f"(counts.dropped.{DUPLICATE_LABEL_TIE}), and the {MEASUREMENT_COUNT_COLUMN} "
    "column records how many source rows each label was aggregated from."
)


class LabelAggregationError(ValueError):
    """Raised when source rows cannot be merged or do not add up."""


# ---------------------------------------------------------------- filtering


@dataclass(frozen=True)
class FilterOutcome:
    """Every source row the SMILES filter kept, and why it dropped the rest."""

    kept_row_indices: list[int]
    isomeric_smiles: list[str]
    dropped: dict[str, int]


def filter_source_smiles(
    raw_smiles: Sequence[str | None], smiles_filter: SmilesFilterConfig
) -> FilterOutcome:
    """Run the shared SMILES filter over the raw SMILES, keeping every occurrence.

    ``smiles_filter.dedupe`` must be off: the merge has to see *all*
    measurements of a compound (:func:`aggregate_by_canonical_smiles`), so the
    collapse happens one step later, and the settings recorded in provenance
    are the ones that ran.

    The filter reports a single ``filtered`` verdict, so the element gate, the
    atom-count gate, the fragment gate and the isotope gate are counted together.

    Raises:
        LabelAggregationError: if ``smiles_filter.dedupe`` is on.
    """
    if smiles_filter.dedupe:
        raise LabelAggregationError(
            "smiles_filter.dedupe must be off for a preparer: duplicates are "
            "merged by aggregate_by_canonical_smiles, which needs every occurrence"
        )
    filtered = filter_smiles(raw_smiles, smiles_filter)
    return FilterOutcome(
        kept_row_indices=filtered.kept_row_indices,
        isomeric_smiles=filtered.isomeric_smiles,
        dropped={INVALID_SMILES: filtered.invalid, SMILES_FILTER: filtered.filtered},
    )


# ------------------------------------------------------------------- merging


def aggregate_label(values: Sequence[float], task_type: TaskType) -> float | None:
    """Collapse the replicate measurements of one compound and one label.

    NaN measurements are ignored. A regression label takes the mean of the
    rest, a classification label their majority vote; an exact tie, or no
    measurement at all, gives ``None``. A compound measured once keeps its
    measurement bit for bit.

    Raises:
        LabelAggregationError: if a classification value is neither 0 nor 1.
            The vote would silently turn such a value into a class, so the 0/1
            invariant ``write_bundle`` checks could no longer see it.
    """
    measured = [value for value in values if not math.isnan(value)]
    if not measured:
        return None
    if task_type is not TaskType.classification:
        return statistics.fmean(measured)
    invalid = sorted({value for value in measured if value not in (0.0, 1.0)})
    if invalid:
        raise LabelAggregationError(
            f"classification measurements {invalid} are neither 0 nor 1"
        )
    positive_votes = sum(1 for value in measured if value == 1.0)
    negative_votes = len(measured) - positive_votes
    if positive_votes == negative_votes:
        return None
    return 1.0 if positive_votes > negative_votes else 0.0


@dataclass(frozen=True)
class AggregatedRows:
    """One row per stereoisomer, with the replicate measurements merged."""

    #: Per output row, the positions into the aggregation input that were
    #: merged into it, in input order; the first is where the row was first seen.
    member_positions: list[list[int]]
    identity: IdentityTable
    #: Label name -> one value per output row (NaN where the label is missing).
    labels: dict[str, list[float]]
    #: Replicate rows merged into an earlier row (``duplicate_smiles``).
    merged_rows: int
    #: Compounds dropped because every label tied exactly.
    label_ties: int
    #: Compounds dropped because the source gave them no label at all.
    missing_labels: int

    @property
    def first_positions(self) -> list[int]:
        return [positions[0] for positions in self.member_positions]

    @property
    def measurement_counts(self) -> list[int]:
        return [len(positions) for positions in self.member_positions]

    @property
    def rows_with_replicates(self) -> int:
        """How many rows carry more than one measurement."""
        return sum(1 for count in self.measurement_counts if count > 1)

    def dropped(self) -> dict[str, int]:
        """The ``counts.dropped`` entries this step is responsible for."""
        return {
            DUPLICATE_SMILES: self.merged_rows,
            DUPLICATE_LABEL_TIE: self.label_ties,
            MISSING_LABEL: self.missing_labels,
        }


def aggregate_by_canonical_smiles(
    isomeric_smiles: Sequence[str],
    labels: Mapping[str, Sequence[float]],
    task_types: Mapping[str, TaskType],
) -> AggregatedRows:
    """Group rows by canonical isomeric SMILES and merge their labels.

    Groups are emitted in first-occurrence order over the input, so a
    compound's identity and row position come from where it was first seen;
    callers order the input by priority (e.g. ``train_val`` before ``test``).

    Raises:
        LabelAggregationError: if a label column's length differs from the
            SMILES, a label has no task type, or a classification value is
            neither 0 nor 1.
    """
    if set(labels) != set(task_types):
        raise LabelAggregationError(
            f"labels {sorted(labels)} and task types {sorted(task_types)} differ"
        )
    for name, values in labels.items():
        if len(values) != len(isomeric_smiles):
            raise LabelAggregationError(
                f"label {name!r} has {len(values)} values for "
                f"{len(isomeric_smiles)} SMILES"
            )

    canonical = [canonical_smiles_pair(smiles).isomeric for smiles in isomeric_smiles]
    positions_by_smiles: dict[str, list[int]] = {}
    for position, smiles in enumerate(canonical):
        positions_by_smiles.setdefault(smiles, []).append(position)

    member_positions: list[list[int]] = []
    aggregated: dict[str, list[float]] = {name: [] for name in labels}
    label_ties = missing_labels = 0
    for positions in positions_by_smiles.values():
        row: dict[str, float | None] = {}
        any_measured = False
        for name, values in labels.items():
            members = [float(values[position]) for position in positions]
            any_measured = any_measured or any(not math.isnan(v) for v in members)
            row[name] = aggregate_label(members, task_types[name])
        if all(value is None for value in row.values()):
            if any_measured:
                label_ties += 1
            else:
                missing_labels += 1
            continue
        member_positions.append(positions)
        for name, value in row.items():
            aggregated[name].append(math.nan if value is None else value)

    return AggregatedRows(
        member_positions=member_positions,
        identity=assign_identity(
            [canonical[positions[0]] for positions in member_positions]
        ),
        labels=aggregated,
        merged_rows=len(canonical) - len(positions_by_smiles),
        label_ties=label_ties,
        missing_labels=missing_labels,
    )


def check_row_accounting(
    source_molecules: int, final_rows: int, dropped: Mapping[str, int]
) -> None:
    """Every source row must be either a bundle row or counted under one reason.

    Raises:
        LabelAggregationError: if the two sides do not add up.
    """
    accounted = final_rows + sum(dropped.values())
    if accounted != source_molecules:
        raise LabelAggregationError(
            f"{final_rows} rows + dropped {dict(dropped)} = {accounted}, which "
            f"does not account for the {source_molecules} source rows"
        )


# ------------------------------------------------------- the whole merge step


@dataclass(frozen=True)
class MergedRows:
    """Source rows filtered and merged into bundle rows, every drop counted."""

    aggregated: AggregatedRows
    #: How many source rows went in.
    source_rows: int
    #: Per bundle row, the source row indices merged into it, in source order.
    member_source_rows: list[list[int]]
    #: ``counts.dropped`` for provenance: the filter's reasons and the merge's.
    dropped: dict[str, int]

    def __len__(self) -> int:
        return len(self.member_source_rows)

    @property
    def first_source_rows(self) -> list[int]:
        """The source row each bundle row was first seen at."""
        return [members[0] for members in self.member_source_rows]

    @property
    def isomeric_smiles(self) -> list[str]:
        return list(self.aggregated.identity.isomeric_smiles)


def merge_source_rows(
    raw_smiles: Sequence[str | None],
    labels: Mapping[str, Sequence[float]],
    task_types: Mapping[str, TaskType],
    smiles_filter: SmilesFilterConfig,
) -> MergedRows:
    """Filter the source rows, merge replicates and check that every row is counted.

    ``labels`` holds one value per *source* row (NaN where missing); the rows
    are taken in the given order, which is the merge priority.

    Raises:
        LabelAggregationError: on mismatched inputs, a classification value
            other than 0 or 1, or rows that do not add up.
    """
    for name, values in labels.items():
        if len(values) != len(raw_smiles):
            raise LabelAggregationError(
                f"label {name!r} has {len(values)} values for {len(raw_smiles)} "
                "source rows"
            )
    filtered = filter_source_smiles(raw_smiles, smiles_filter)
    kept = filtered.kept_row_indices
    aggregated = aggregate_by_canonical_smiles(
        filtered.isomeric_smiles,
        {name: [values[index] for index in kept] for name, values in labels.items()},
        task_types,
    )
    dropped = {**filtered.dropped, **aggregated.dropped()}
    check_row_accounting(len(raw_smiles), len(aggregated.member_positions), dropped)
    return MergedRows(
        aggregated=aggregated,
        source_rows=len(raw_smiles),
        member_source_rows=[
            [kept[position] for position in positions]
            for positions in aggregated.member_positions
        ],
        dropped=dropped,
    )
