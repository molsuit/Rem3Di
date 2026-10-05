"""The source-row merge every SMILES preparer shares (``bundle/aggregation.py``)."""

from __future__ import annotations

import math

import pytest

from remedi.data_handling.bundle import (
    DUPLICATE_LABEL_TIE,
    DUPLICATE_SMILES,
    MISSING_LABEL,
    AggregatedRows,
    LabelAggregationError,
    aggregate_by_canonical_smiles,
    aggregate_label,
    check_row_accounting,
    filter_source_smiles,
    merge_source_rows,
)
from remedi.data_handling.chemistry.smiles_filter import SmilesFilterConfig
from remedi.data_handling.dataset.tasks import TaskType

ETHANOL = "CCO"
PROPANOL = "CCCO"
L_ALANINE = "C[C@@H](N)C(=O)O"
D_ALANINE = "C[C@H](N)C(=O)O"
NAN = math.nan
#: Preparers run the filter without dedupe: the merge collapses duplicates.
PREPARER_FILTER = SmilesFilterConfig(dedupe=False)


def _single(
    smiles: list[str], labels: list[float], task_type: TaskType
) -> AggregatedRows:
    return aggregate_by_canonical_smiles(smiles, {"y": labels}, {"y": task_type})


# ---------------------------------------------------------------- filtering


def test_filter_counts_every_drop_reason_and_keeps_every_occurrence() -> None:
    raw = [
        ETHANOL,  # kept
        "not a molecule",  # invalid
        "[11CH3]CO",  # isotope -> filtered
        ETHANOL,  # a replicate, kept for aggregation
        PROPANOL,  # kept
    ]
    outcome = filter_source_smiles(raw, PREPARER_FILTER)
    assert outcome.dropped == {"invalid_smiles": 1, "smiles_filter": 1}
    assert outcome.kept_row_indices == [0, 3, 4]
    assert outcome.isomeric_smiles == [ETHANOL, ETHANOL, PROPANOL]


def test_filter_refuses_dedupe_because_the_merge_needs_every_occurrence() -> None:
    with pytest.raises(LabelAggregationError, match="dedupe must be off"):
        filter_source_smiles([ETHANOL], SmilesFilterConfig(dedupe=True))


def test_filter_strips_the_counterion_of_a_salt() -> None:
    """Salt stripping is on, so the counter-ion is removed instead of dropped."""
    outcome = filter_source_smiles(["CCO.[Cl-]"], PREPARER_FILTER)
    assert outcome.isomeric_smiles == ["CCO"]


# ------------------------------------------------------------ one label


REGRESSION, CLASSIFICATION = TaskType.regression, TaskType.classification


@pytest.mark.parametrize(
    ("values", "task_type", "expected"),
    [
        ([1.0, 2.0, 3.0], REGRESSION, 2.0),  # mean
        ([2.5], REGRESSION, 2.5),
        ([1.0, 1.0, 0.0], CLASSIFICATION, 1.0),  # majority vote
        ([0.0, 0.0, 1.0], CLASSIFICATION, 0.0),
        ([1.0], CLASSIFICATION, 1.0),
        ([0.0, 1.0], CLASSIFICATION, None),  # exact tie
        ([1.0, 0.0, 1.0, 0.0], CLASSIFICATION, None),
        ([NAN, 2.0, 4.0], REGRESSION, 3.0),  # missing measurements are ignored
        ([NAN, 1.0], CLASSIFICATION, 1.0),
        ([NAN, NAN], REGRESSION, None),
        ([], REGRESSION, None),
    ],
)
def test_aggregate_label(
    values: list[float], task_type: TaskType, expected: float | None
) -> None:
    assert aggregate_label(values, task_type) == expected


def test_aggregate_label_rejects_a_classification_value_outside_zero_one() -> None:
    with pytest.raises(LabelAggregationError, match="neither 0 nor 1"):
        aggregate_label([1.0, 2.0], CLASSIFICATION)


# ------------------------------------------------------ merging rows, one label


def test_aggregation_means_the_replicates_of_a_regression_label() -> None:
    aggregated = _single(
        [ETHANOL, PROPANOL, "OCC", ETHANOL], [1.0, 7.0, 2.0, 6.0], TaskType.regression
    )
    # ethanol is first seen at position 0, propanol at position 1.
    assert aggregated.member_positions == [[0, 2, 3], [1]]
    assert aggregated.labels == {"y": [3.0, 7.0]}
    assert aggregated.measurement_counts == [3, 1]
    assert aggregated.merged_rows == 2
    assert aggregated.rows_with_replicates == 1
    assert aggregated.dropped() == {
        DUPLICATE_SMILES: 2,
        DUPLICATE_LABEL_TIE: 0,
        MISSING_LABEL: 0,
    }


def test_aggregation_drops_a_tied_compound_and_counts_it() -> None:
    aggregated = _single(
        [ETHANOL, ETHANOL, PROPANOL], [1.0, 0.0, 1.0], TaskType.classification
    )
    assert aggregated.member_positions == [[2]]
    assert aggregated.labels == {"y": [1.0]}
    assert aggregated.merged_rows == 1
    assert aggregated.label_ties == 1


def test_aggregation_keeps_enantiomers_apart() -> None:
    aggregated = _single([L_ALANINE, D_ALANINE], [0.0, 1.0], TaskType.classification)
    assert aggregated.member_positions == [[0], [1]]
    assert aggregated.merged_rows == 0
    assert list(aggregated.identity.molecule_id) == [0, 0]
    assert list(aggregated.identity.enantiomer_of) == [1, 0]


# ---------------------------------------------------- merging rows, many labels


def test_aggregation_merges_each_label_column_on_its_own() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [ETHANOL, ETHANOL, PROPANOL],
        {"a": [1.0, NAN, 0.0], "b": [NAN, 3.0, NAN]},
        {"a": TaskType.classification, "b": TaskType.regression},
    )
    assert aggregated.labels["a"] == [1.0, 0.0]
    assert aggregated.labels["b"][0] == 3.0
    assert math.isnan(aggregated.labels["b"][1])


def test_a_tie_in_one_label_keeps_the_row_for_the_others() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [ETHANOL, ETHANOL],
        {"a": [1.0, 0.0], "b": [2.0, 4.0]},
        {"a": TaskType.classification, "b": TaskType.regression},
    )
    assert math.isnan(aggregated.labels["a"][0])
    assert aggregated.labels["b"] == [3.0]
    assert aggregated.label_ties == 0


def test_a_compound_without_any_label_is_dropped_as_missing() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [ETHANOL, PROPANOL],
        {"a": [1.0, NAN], "b": [NAN, NAN]},
        {"a": TaskType.classification, "b": TaskType.regression},
    )
    assert aggregated.member_positions == [[0]]
    assert aggregated.missing_labels == 1
    assert aggregated.label_ties == 0


def test_aggregation_rejects_mismatched_inputs() -> None:
    with pytest.raises(LabelAggregationError, match="values for"):
        aggregate_by_canonical_smiles(
            [ETHANOL], {"a": [1.0, 2.0]}, {"a": TaskType.regression}
        )
    with pytest.raises(LabelAggregationError, match="task types"):
        aggregate_by_canonical_smiles([ETHANOL], {"a": [1.0]}, {})
    with pytest.raises(LabelAggregationError, match="source rows"):
        merge_source_rows(
            [ETHANOL, PROPANOL], {"a": [1.0]}, {"a": REGRESSION}, PREPARER_FILTER
        )


def test_row_accounting_adds_up_or_raises() -> None:
    check_row_accounting(10, 7, {"invalid_smiles": 1, DUPLICATE_SMILES: 2})
    with pytest.raises(LabelAggregationError, match="does not account for"):
        check_row_accounting(10, 7, {"invalid_smiles": 1})


# ------------------------------------------------------- the whole merge step


def test_merge_maps_bundle_rows_back_to_source_rows_and_counts_every_drop() -> None:
    raw = [ETHANOL, "not a molecule", PROPANOL, "OCC", "[11CH3]CO", L_ALANINE]
    merged = merge_source_rows(
        raw,
        {"y": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0]},
        {"y": TaskType.regression},
        PREPARER_FILTER,
    )
    assert len(merged) == 3
    assert merged.source_rows == 6
    # ethanol (rows 0 and 3), propanol (row 2), alanine (row 5)
    assert merged.member_source_rows == [[0, 3], [2], [5]]
    assert merged.first_source_rows == [0, 2, 5]
    assert merged.isomeric_smiles == [ETHANOL, PROPANOL, L_ALANINE]
    assert merged.aggregated.labels == {"y": [3.0, 3.0, 6.0]}
    assert merged.dropped == {
        "invalid_smiles": 1,
        "smiles_filter": 1,
        DUPLICATE_SMILES: 1,
        DUPLICATE_LABEL_TIE: 0,
        MISSING_LABEL: 0,
    }


def test_rows_dropped_before_the_merge_are_counted_first() -> None:
    merged = merge_source_rows(
        [ETHANOL, PROPANOL],
        {"y": [1.0, 2.0]},
        {"y": TaskType.regression},
        PREPARER_FILTER,
    ).after_source_drops({"or_stereo_group": 3})
    assert merged.source_rows == 5
    assert next(iter(merged.dropped)) == "or_stereo_group"
    assert merged.member_source_rows == [[0], [1]]
    with pytest.raises(LabelAggregationError, match="counted twice"):
        merged.after_source_drops({DUPLICATE_SMILES: 1})
