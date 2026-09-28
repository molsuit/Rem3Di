"""Tests for the TDC ADMET preparer.

The pure functions are exercised on a tiny synthetic frame; the end-to-end test
runs the smallest real endpoint and is skipped when the pinned ``admet_group``
download is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from remedi_prepare_tdc.prepare import (
    AGGREGATION_NOTICE,
    DUPLICATE_LABEL_TIE,
    DUPLICATE_SMILES,
    MEASUREMENT_COUNT_COLUMN,
    TDC_ENDPOINTS,
    AggregatedRows,
    SourceTable,
    TdcPreparationError,
    TdcPreparerConfig,
    aggregate_by_canonical_smiles,
    aggregate_label,
    build_spec,
    build_table,
    check_classification_labels,
    check_row_accounting,
    filter_source_smiles,
    headline_metric,
    load_admet_group,
    metrics_for_endpoint,
    prepare_endpoint,
    preparer_record,
    read_source_table,
    split_column_values,
    task_type_for_metric,
    tdc_default_metrics,
    train_valid_labels,
)

from remedi.configuration.dataset_config import FilterMoleculeStageConfig
from remedi.data_handling.bundle import (
    EvalMetric,
    SmilesFilterRecord,
    read_bundle,
)
from remedi.data_handling.dataset.tasks import TaskType

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "tdc"

ETHANOL = "CCO"
PROPANOL = "CCCO"
BUTANOL = "CCCCO"
L_ALANINE = "C[C@@H](N)C(=O)O"
D_ALANINE = "C[C@H](N)C(=O)O"


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Drug_ID", "Drug", "Y"])


# ------------------------------------------------------- metrics and tasks


def test_headline_metric_maps_every_tdc_metric() -> None:
    assert headline_metric("mae") is EvalMetric.mae
    assert headline_metric("spearman") is EvalMetric.spearman
    assert headline_metric("roc-auc") is EvalMetric.auroc
    assert headline_metric("pr-auc") is EvalMetric.auprc


def test_headline_metric_rejects_an_unknown_metric() -> None:
    with pytest.raises(TdcPreparationError, match="no EvalMetric equivalent"):
        headline_metric("accuracy")


def test_task_type_follows_the_metric_family() -> None:
    assert task_type_for_metric(EvalMetric.auroc) is TaskType.classification
    assert task_type_for_metric(EvalMetric.auprc) is TaskType.classification
    assert task_type_for_metric(EvalMetric.mae) is TaskType.regression
    assert task_type_for_metric(EvalMetric.spearman) is TaskType.regression


def test_metric_list_puts_the_tdc_default_first_and_deduplicates() -> None:
    assert metrics_for_endpoint("roc-auc") == [EvalMetric.auroc, EvalMetric.auprc]
    assert metrics_for_endpoint("pr-auc") == [EvalMetric.auprc, EvalMetric.auroc]
    assert metrics_for_endpoint("mae") == [
        EvalMetric.mae,
        EvalMetric.rmse,
        EvalMetric.spearman,
        EvalMetric.r2,
    ]
    assert metrics_for_endpoint("spearman") == [
        EvalMetric.spearman,
        EvalMetric.mae,
        EvalMetric.rmse,
        EvalMetric.r2,
    ]


def test_every_endpoint_has_a_unique_id_and_task_name() -> None:
    dataset_ids = [endpoint.dataset_id for endpoint in TDC_ENDPOINTS]
    task_names = [endpoint.task_name for endpoint in TDC_ENDPOINTS]
    assert len(TDC_ENDPOINTS) == 22
    assert len(set(dataset_ids)) == 22
    assert len(set(task_names)) == 22


# ------------------------------------------------------------- the config


def test_split_columns_alias_the_first_seed() -> None:
    config = TdcPreparerConfig(seeds=[1, 2, 3])
    assert config.split_columns() == [
        "split",
        "split__seed1",
        "split__seed2",
        "split__seed3",
    ]


def test_config_rejects_an_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="unknown dataset ids"):
        TdcPreparerConfig(only=["Not_A_Benchmark"])


def test_smiles_filter_record_mirrors_the_stage_config() -> None:
    config = TdcPreparerConfig()
    record = config.smiles_filter_record()
    stage_fields = set(FilterMoleculeStageConfig.model_fields) - {"kind"}
    assert set(SmilesFilterRecord.model_fields) == stage_fields
    assert record.max_atoms == 100
    assert record.strip_salts is True
    assert record.neutralize is True
    assert record.dedupe is True


# -------------------------------------------------------- the split columns


def test_train_valid_labels_recovers_the_partition() -> None:
    train_val = _frame([("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0), ("c", BUTANOL, 0.0)])
    train = _frame([("c", BUTANOL, 0.0), ("a", ETHANOL, 0.0)])
    valid = _frame([("b", PROPANOL, 1.0)])
    assert train_valid_labels(train_val, train, valid) == ["train", "valid", "train"]


def test_train_valid_labels_gives_identical_rows_train_first() -> None:
    """Identical rows are interchangeable; the earlier position keeps ``train``."""
    train_val = _frame([("a", ETHANOL, 0.0), ("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)])
    train = _frame([("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)])
    valid = _frame([("a", ETHANOL, 0.0)])
    assert train_valid_labels(train_val, train, valid) == ["train", "valid", "train"]


def test_train_valid_labels_rejects_an_unassigned_row() -> None:
    train_val = _frame([("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)])
    train = _frame([("a", ETHANOL, 0.0)])
    valid = pd.DataFrame(columns=["Drug_ID", "Drug", "Y"])
    with pytest.raises(TdcPreparationError, match="neither the train nor the valid"):
        train_valid_labels(train_val, train, valid)


def test_train_valid_labels_rejects_a_foreign_row() -> None:
    train_val = _frame([("a", ETHANOL, 0.0)])
    train = _frame([("z", BUTANOL, 1.0)])
    valid = pd.DataFrame(columns=["Drug_ID", "Drug", "Y"])
    with pytest.raises(TdcPreparationError, match="not an unclaimed train_val row"):
        train_valid_labels(train_val, train, valid)


def test_split_column_values_appends_the_fixed_test_fold() -> None:
    source = SourceTable(
        train_val=_frame([("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)]),
        test=_frame([("c", BUTANOL, 1.0)]),
    )
    train = _frame([("a", ETHANOL, 0.0)])
    valid = _frame([("b", PROPANOL, 1.0)])
    assert split_column_values(source, train, valid) == ["train", "valid", "test"]


def test_split_columns_differ_across_seeds_but_cover_every_row() -> None:
    source = SourceTable(
        train_val=_frame(
            [("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0), ("c", BUTANOL, 0.0)]
        ),
        test=_frame([("d", L_ALANINE, 1.0)]),
    )
    per_seed = {
        "split__seed1": split_column_values(
            source,
            _frame([("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)]),
            _frame([("c", BUTANOL, 0.0)]),
        ),
        "split__seed2": split_column_values(
            source,
            _frame([("b", PROPANOL, 1.0), ("c", BUTANOL, 0.0)]),
            _frame([("a", ETHANOL, 0.0)]),
        ),
    }
    assert per_seed["split__seed1"] == ["train", "train", "valid", "test"]
    assert per_seed["split__seed2"] == ["valid", "train", "train", "test"]
    for values in per_seed.values():
        assert len(values) == len(source)
        assert values[-1] == "test"


# -------------------------------------------------- filtering + aggregation


def test_filter_counts_every_drop_reason_and_keeps_every_occurrence() -> None:
    raw = [
        ETHANOL,  # kept
        "not a molecule",  # invalid
        "[11CH3]CO",  # isotope -> filtered
        ETHANOL,  # a replicate, kept for aggregation
        PROPANOL,  # kept
    ]
    outcome = filter_source_smiles(raw, FilterMoleculeStageConfig())
    assert outcome.dropped == {"invalid_smiles": 1, "smiles_filter": 1}
    assert outcome.kept_row_indices == [0, 3, 4]
    assert outcome.isomeric_smiles == [ETHANOL, ETHANOL, PROPANOL]


def test_filter_drops_a_multifragment_salt_free_of_its_counterion() -> None:
    """Salt stripping is on, so the counter-ion is removed instead of dropped."""
    outcome = filter_source_smiles(["CCO.[Cl-]"], FilterMoleculeStageConfig())
    assert outcome.isomeric_smiles == ["CCO"]


def test_aggregate_label_averages_a_regression_endpoint() -> None:
    assert aggregate_label([1.0, 2.0, 3.0], TaskType.regression) == 2.0
    assert aggregate_label([2.5], TaskType.regression) == 2.5


def test_aggregate_label_takes_a_classification_majority() -> None:
    assert aggregate_label([1.0, 1.0, 0.0], TaskType.classification) == 1.0
    assert aggregate_label([0.0, 0.0, 1.0], TaskType.classification) == 0.0
    assert aggregate_label([1.0], TaskType.classification) == 1.0


def test_aggregate_label_returns_none_on_an_exact_tie() -> None:
    assert aggregate_label([0.0, 1.0], TaskType.classification) is None
    assert aggregate_label([1.0, 0.0, 1.0, 0.0], TaskType.classification) is None


def test_aggregate_label_rejects_no_measurements() -> None:
    with pytest.raises(TdcPreparationError, match="empty list of measurements"):
        aggregate_label([], TaskType.regression)


def test_aggregation_means_the_replicates_of_a_regression_endpoint() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [ETHANOL, PROPANOL, "OCC", ETHANOL],
        [1.0, 7.0, 2.0, 6.0],
        TaskType.regression,
    )
    assert isinstance(aggregated, AggregatedRows)
    # ethanol is first seen at position 0, propanol at position 1.
    assert aggregated.first_positions == [0, 1]
    assert aggregated.labels == [3.0, 7.0]
    assert aggregated.measurement_counts == [3, 1]
    assert aggregated.merged_rows == 2
    assert aggregated.label_ties == 0


def test_aggregation_majority_votes_a_classification_endpoint() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [ETHANOL, ETHANOL, ETHANOL, PROPANOL],
        [1.0, 0.0, 1.0, 0.0],
        TaskType.classification,
    )
    assert aggregated.labels == [1.0, 0.0]
    assert aggregated.measurement_counts == [3, 1]
    assert aggregated.merged_rows == 2
    assert aggregated.label_ties == 0


def test_aggregation_drops_a_tied_compound_and_counts_it() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [ETHANOL, ETHANOL, PROPANOL],
        [1.0, 0.0, 1.0],
        TaskType.classification,
    )
    assert aggregated.first_positions == [2]
    assert aggregated.labels == [1.0]
    assert aggregated.measurement_counts == [1]
    assert aggregated.merged_rows == 1
    assert aggregated.label_ties == 1


def test_aggregation_keeps_the_first_occurrence_position_for_the_split() -> None:
    """The kept row is the first occurrence, so it carries the earliest split."""
    aggregated = aggregate_by_canonical_smiles(
        [PROPANOL, ETHANOL, "OCCC"], [1.0, 2.0, 3.0], TaskType.regression
    )
    assert aggregated.first_positions == [0, 1]
    assert aggregated.labels == [2.0, 2.0]
    assert aggregated.measurement_counts == [2, 1]


def test_aggregation_keeps_enantiomers_apart() -> None:
    aggregated = aggregate_by_canonical_smiles(
        [L_ALANINE, D_ALANINE], [0.0, 1.0], TaskType.classification
    )
    assert aggregated.first_positions == [0, 1]
    assert aggregated.merged_rows == 0
    assert list(aggregated.identity.stereoisomer_id) == [0, 1]
    assert list(aggregated.identity.molecule_id) == [0, 0]
    assert list(aggregated.identity.enantiomer_of) == [1, 0]


def test_row_accounting_adds_up_or_raises() -> None:
    check_row_accounting(10, 7, {"invalid_smiles": 1, DUPLICATE_SMILES: 2})
    with pytest.raises(TdcPreparationError, match="does not account for"):
        check_row_accounting(10, 7, {"invalid_smiles": 1})


# ------------------------------------------------------------- table + spec


def _identity(smiles: list[str], labels: list[float]) -> AggregatedRows:
    return aggregate_by_canonical_smiles(smiles, labels, TaskType.regression)


def test_build_table_has_the_declared_columns_in_order() -> None:
    aggregated = _identity([ETHANOL, PROPANOL], [0.0, 1.0])
    spec = build_spec(TDC_ENDPOINTS[1], "roc-auc", ["split", "split__seed1"])
    table = build_table(
        identity=aggregated.identity,
        task_name=spec.tasks[0].name,
        labels=[0.0, 1.0],
        split_values={"split": ["train", "test"], "split__seed1": ["train", "test"]},
        measurement_counts=[3, 1],
    )
    assert list(table.columns) == spec.expected_columns()
    assert list(table.columns)[-1] == MEASUREMENT_COUNT_COLUMN
    assert str(table[spec.tasks[0].name].dtype) == "float64"
    assert str(table[MEASUREMENT_COUNT_COLUMN].dtype) == "float64"
    assert list(table[MEASUREMENT_COUNT_COLUMN]) == [3.0, 1.0]


def test_build_table_rejects_a_missing_label() -> None:
    aggregated = _identity([ETHANOL], [0.0])
    with pytest.raises(TdcPreparationError, match="NaN labels"):
        build_table(
            identity=aggregated.identity,
            task_name="HIA",
            labels=[float("nan")],
            split_values={"split": ["train"]},
            measurement_counts=[1],
        )


def test_build_table_rejects_an_unassigned_split() -> None:
    aggregated = _identity([ETHANOL], [0.0])
    with pytest.raises(TdcPreparationError, match="unassigned rows"):
        build_table(
            identity=aggregated.identity,
            task_name="HIA",
            labels=[1.0],
            split_values={"split": ["unassigned"]},
            measurement_counts=[1],
        )


def test_build_table_rejects_a_zero_measurement_count() -> None:
    aggregated = _identity([ETHANOL], [0.0])
    with pytest.raises(TdcPreparationError, match="must be at least 1"):
        build_table(
            identity=aggregated.identity,
            task_name="HIA",
            labels=[1.0],
            split_values={"split": ["train"]},
            measurement_counts=[0],
        )


def test_build_spec_is_a_smiles_stage_stereoisomer_split_bundle() -> None:
    spec = build_spec(TDC_ENDPOINTS[0], "mae", ["split", "split__seed1"])
    assert spec.stage == "smiles"
    assert spec.geometry_origin is None
    assert spec.split_group == "stereoisomer_id"
    assert spec.require_enantiomer_pairs is False
    assert spec.extra_columns == [MEASUREMENT_COUNT_COLUMN]
    assert spec.source_kind == "tdc_admet_group"
    assert spec.default_split == "split"
    assert spec.tasks[0].task_type is TaskType.regression
    assert "Caco2_Wang" in spec.description


def test_check_classification_labels() -> None:
    check_classification_labels("HIA", [0.0, 1.0, 1.0])
    with pytest.raises(TdcPreparationError, match=r"expected only 0\.0 and 1\.0"):
        check_classification_labels("HIA", [0.0, 2.0])


# ------------------------------------------------------------- end to end


@pytest.mark.skipif(
    not (RAW_ROOT / "admet_group").is_dir(),
    reason="the pinned PyTDC admet_group download is not present",
)
def test_prepare_the_smallest_endpoint_end_to_end(tmp_path: Path) -> None:
    endpoint = next(
        item for item in TDC_ENDPOINTS if item.dataset_id == "Bioavailability_Ma"
    )
    config = TdcPreparerConfig(
        raw_root=RAW_ROOT,
        bundle_root=tmp_path / "bundles",
        seeds=[1, 2],
        only=[endpoint.dataset_id],
    )
    group = load_admet_group(config.raw_root)
    metrics = tdc_default_metrics()
    report = prepare_endpoint(
        endpoint=endpoint,
        config=config,
        group=group,
        tdc_metric=metrics["bioavailability_ma"],
        preparer=preparer_record(REPOSITORY_ROOT),
    )

    source = read_source_table(RAW_ROOT, "bioavailability_ma")
    assert report.source_molecules == len(source)
    assert report.final_rows + sum(report.dropped.values()) == report.source_molecules
    assert set(report.dropped) == {
        "invalid_smiles",
        "smiles_filter",
        DUPLICATE_SMILES,
        DUPLICATE_LABEL_TIE,
    }
    assert set(report.per_split) == {"train", "valid", "test"}
    assert report.per_split["test"] > 0
    assert not config.staging_root().exists()

    bundle = read_bundle(config.bundle_root / endpoint.dataset_id)
    assert bundle.spec.split_columns == ["split", "split__seed1", "split__seed2"]
    assert bundle.spec.metrics[0] is EvalMetric.auroc
    assert bundle.spec.extra_columns == [MEASUREMENT_COUNT_COLUMN]
    assert bundle.structures is None
    assert not (config.bundle_root / endpoint.dataset_id / "structures.extxyz").exists()
    assert (bundle.table["split"] == bundle.table["split__seed1"]).all()
    assert set(bundle.table["split"]) <= {"train", "valid", "test"}
    assert bundle.table[endpoint.task_name].notna().all()
    assert set(bundle.table[endpoint.task_name].unique()) <= {0.0, 1.0}
    counts = bundle.table[MEASUREMENT_COUNT_COLUMN]
    assert (counts >= 1.0).all()
    assert len(counts) == report.final_rows
    assert bundle.provenance.notices == [AGGREGATION_NOTICE]
    assert bundle.provenance.smiles_filter is not None
    assert bundle.provenance.smiles_filter.max_atoms == 100
    assert set(bundle.provenance.source.package_versions) == {"PyTDC", "rdkit"}
    assert len(bundle.provenance.source.files) == 2
    assert bundle.provenance.counts.source_molecules == len(source)


@pytest.mark.skipif(
    not (RAW_ROOT / "admet_group").is_dir(),
    reason="the pinned PyTDC admet_group download is not present",
)
def test_replicates_are_aggregated_on_a_real_endpoint(tmp_path: Path) -> None:
    """PPBR_AZ is the endpoint with the most replicate measurements."""
    endpoint = next(item for item in TDC_ENDPOINTS if item.dataset_id == "PPBR_AZ")
    config = TdcPreparerConfig(
        raw_root=RAW_ROOT,
        bundle_root=tmp_path / "bundles",
        seeds=[1],
        only=[endpoint.dataset_id],
    )
    report = prepare_endpoint(
        endpoint=endpoint,
        config=config,
        group=load_admet_group(config.raw_root),
        tdc_metric=tdc_default_metrics()["ppbr_az"],
        preparer=preparer_record(REPOSITORY_ROOT),
    )
    assert report.dropped[DUPLICATE_SMILES] > 0
    assert report.dropped[DUPLICATE_LABEL_TIE] == 0  # a regression endpoint never ties
    assert report.aggregated_rows > 0

    table = read_bundle(config.bundle_root / endpoint.dataset_id).table
    counts = table[MEASUREMENT_COUNT_COLUMN]
    # every source row that survived the filter went into exactly one label
    assert counts.sum() == report.final_rows + report.dropped[DUPLICATE_SMILES]
    assert counts.max() > 1.0
