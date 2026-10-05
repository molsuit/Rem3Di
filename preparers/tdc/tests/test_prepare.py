"""Tests for the TDC ADMET preparer.

The pure functions are exercised on tiny synthetic frames, ``prepare_endpoint``
runs end to end on synthetic csvs with a stand-in for the PyTDC benchmark group,
and two tests run real endpoints, skipped when the pinned ``admet_group``
download is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from remedi_prepare_tdc.prepare import (
    ENDPOINTS_FILE,
    TDC_ENDPOINTS,
    SourceTable,
    TdcEndpoint,
    TdcEndpointCatalog,
    TdcPreparationError,
    TdcPreparerConfig,
    build_spec,
    headline_metric,
    load_admet_group,
    main,
    prepare_endpoint,
    raw_split_columns,
    read_source_table,
    source_file_hashes,
    split_column_values,
    task_type_for_metric,
    tdc_default_metrics,
    train_valid_labels,
)

from remedi.data_handling.bundle import (
    AGGREGATION_NOTICE,
    DUPLICATE_LABEL_TIE,
    DUPLICATE_SMILES,
    MEASUREMENT_COUNT_COLUMN,
    MISSING_LABEL,
    BundleValidationError,
    EvalMetric,
    LabelAggregationError,
    PreparerRecord,
    read_bundle,
)
from remedi.data_handling.bundle.preparation import format_report
from remedi.data_handling.dataset.tasks import TaskType

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "tdc"
PREPARER = PreparerRecord(repo="molsuit/Rem3Di", script="prepare.py")

requires_raw_download = pytest.mark.skipif(
    not (RAW_ROOT / "admet_group").is_dir(),
    reason="the pinned PyTDC admet_group download is not present",
)

ETHANOL = "CCO"
PROPANOL = "CCCO"
BUTANOL = "CCCCO"
L_ALANINE = "C[C@@H](N)C(=O)O"
D_ALANINE = "C[C@H](N)C(=O)O"


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Drug_ID", "Drug", "Y"])


def _endpoint(dataset_id: str) -> TdcEndpoint:
    return next(item for item in TDC_ENDPOINTS if item.dataset_id == dataset_id)


# ------------------------------------------------------- the endpoint table


def test_the_catalog_lists_22_endpoints_with_unique_names() -> None:
    dataset_ids = [endpoint.dataset_id for endpoint in TDC_ENDPOINTS]
    label_names = [endpoint.label_name for endpoint in TDC_ENDPOINTS]
    assert len(TDC_ENDPOINTS) == 22
    assert len(set(dataset_ids)) == 22
    assert len(set(label_names)) == 22
    assert TdcEndpointCatalog.from_yaml(ENDPOINTS_FILE).dataset_ids() == dataset_ids


def test_the_catalog_rejects_a_duplicate_dataset_id(tmp_path: Path) -> None:
    entry = {"dataset_id": "A", "label_name": "a", "property_description": "x"}
    other = {"dataset_id": "A", "label_name": "b", "property_description": "y"}
    path = tmp_path / "endpoints.yaml"
    path.write_text(yaml.safe_dump({"endpoints": [entry, other]}))
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        TdcEndpointCatalog.from_yaml(path)


def test_the_catalog_rejects_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="task_name"):
        TdcEndpointCatalog.model_validate(
            {
                "endpoints": [
                    {
                        "dataset_id": "A",
                        "task_name": "a",
                        "property_description": "x",
                    }
                ]
            }
        )


# ------------------------------------------------------------------ metrics


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


def test_task_type_rejects_a_metric_outside_every_family() -> None:
    with pytest.raises(TdcPreparationError, match="belongs to no task type"):
        task_type_for_metric(EvalMetric.macro_f1)


# ------------------------------------------------------------- the config


def test_config_rejects_an_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="unknown dataset ids"):
        TdcPreparerConfig(only=["Not_A_Benchmark"])


def test_config_selects_endpoints_in_catalog_order() -> None:
    assert len(TdcPreparerConfig().selected_endpoints()) == 22
    selected = TdcPreparerConfig(only=["DILI", "HIA_Hou"]).selected_endpoints()
    assert [endpoint.dataset_id for endpoint in selected] == ["HIA_Hou", "DILI"]


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


class FakeAdmetGroup:
    """Stands in for PyTDC's ``admet_group``: one endpoint, one partition per seed.

    Seed ``s`` puts train_val row ``s mod n`` into valid and the rest into train.
    """

    def __init__(self, directory_name: str, train_val: pd.DataFrame):
        self.dataset_names = [directory_name]
        self.train_val = train_val

    def get_train_valid_split(
        self, seed: int, benchmark: str, split_type: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert split_type == "default"
        valid_position = seed % len(self.train_val)
        valid_mask = [
            position == valid_position for position in range(len(self.train_val))
        ]
        is_valid = pd.Series(valid_mask)
        train = self.train_val[~is_valid.to_numpy()].reset_index(drop=True)
        valid = self.train_val[is_valid.to_numpy()].reset_index(drop=True)
        return train, valid


def test_raw_split_columns_alias_the_first_seed_and_differ_across_seeds() -> None:
    source = SourceTable(
        train_val=_frame(
            [("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0), ("c", BUTANOL, 0.0)]
        ),
        test=_frame([("d", L_ALANINE, 1.0)]),
    )
    group = FakeAdmetGroup("hia_hou", source.train_val)
    columns = raw_split_columns(group, "HIA_Hou", source, [2, 1])
    assert list(columns) == ["split", "split__seed2", "split__seed1"]
    assert columns["split"] == columns["split__seed2"]
    assert columns["split__seed2"] == ["train", "train", "valid", "test"]
    assert columns["split__seed1"] == ["train", "valid", "train", "test"]


# ------------------------------------------------------------- table + spec


def test_build_spec_is_a_smiles_bundle_split_by_stereoisomer() -> None:
    spec = build_spec(_endpoint("Caco2_Wang"), "mae", ["split", "split__seed1"])
    assert spec.smiles is True
    assert spec.geometry_origin is None
    assert not spec.has_structures
    assert spec.extra_columns == [MEASUREMENT_COUNT_COLUMN]
    assert spec.source_kind == "tdc_admet_group"
    assert [label.name for label in spec.labels] == ["Caco-2"]
    assert spec.labels[0].task_type is TaskType.regression
    assert "Caco2_Wang" in spec.description
    evaluation = spec.evaluation
    assert evaluation is not None
    assert evaluation.metrics == [
        EvalMetric.mae,
        EvalMetric.rmse,
        EvalMetric.spearman,
        EvalMetric.r2,
    ]
    assert evaluation.split_columns == ["split", "split__seed1"]
    assert evaluation.default_split == "split"
    assert evaluation.split_group == "stereoisomer_id"
    assert evaluation.require_enantiomer_pairs is False


def test_build_spec_puts_the_tdc_headline_first() -> None:
    spearman = build_spec(_endpoint("PPBR_AZ"), "spearman", ["split"]).evaluation
    pr_auc = build_spec(_endpoint("DILI"), "pr-auc", ["split"])
    assert spearman is not None and pr_auc.evaluation is not None
    assert spearman.metrics[0] is EvalMetric.spearman
    assert pr_auc.evaluation.metrics == [EvalMetric.auprc, EvalMetric.auroc]
    assert pr_auc.labels[0].task_type is TaskType.classification


# ------------------------------------------- end to end on synthetic csvs


def _write_raw_endpoint(
    raw_root: Path,
    directory_name: str,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    directory = raw_root / "admet_group" / directory_name
    directory.mkdir(parents=True)
    train_val.to_csv(directory / "train_val.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)


def test_prepare_endpoint_writes_a_valid_bundle(tmp_path: Path) -> None:
    train_val = _frame(
        [
            ("a", ETHANOL, 1.0),
            ("b", PROPANOL, 0.0),
            ("c", "not a molecule", 1.0),
            ("d", "OCC", 1.0),  # ethanol again: merged, majority 1
            ("e", L_ALANINE, 0.0),
            ("f", BUTANOL, 1.0),
            ("g", BUTANOL, 0.0),  # ties with f: butanol dropped
        ]
    )
    test = _frame([("h", D_ALANINE, 1.0), ("i", "[11CH3]CO", 0.0)])
    raw_root = tmp_path / "raw"
    _write_raw_endpoint(raw_root, "hia_hou", train_val, test)
    config = TdcPreparerConfig(
        raw_root=raw_root,
        bundle_root=tmp_path / "bundles",
        seeds=[1, 2],
        only=["HIA_Hou"],
    )

    report = prepare_endpoint(
        endpoint=_endpoint("HIA_Hou"),
        config=config,
        group=FakeAdmetGroup("hia_hou", train_val),
        tdc_metric="roc-auc",
        preparer=PREPARER,
    )

    assert report.counts.source_molecules == 9
    assert report.counts.dropped == {
        "invalid_smiles": 1,
        "smiles_filter": 1,
        DUPLICATE_SMILES: 2,
        DUPLICATE_LABEL_TIE: 1,
        MISSING_LABEL: 0,
    }
    assert report.counts.final_rows == 4
    assert report.counts.per_label_non_null == {"HIA": 4}
    assert report.rows_with_replicates == 1
    assert report.spec.evaluation is not None
    assert report.spec.evaluation.metrics[0] is EvalMetric.auroc
    assert report.spec.labels[0].task_type is TaskType.classification

    bundle = read_bundle(report.directory)
    assert bundle.provenance.outputs is not None
    assert (
        report.content_sha256 == bundle.provenance.outputs.table_parquet.content_sha256
    )
    table = bundle.table
    assert "structure_id" not in table.columns
    assert list(table["isomeric_smiles"]) == [ETHANOL, PROPANOL, L_ALANINE, D_ALANINE]
    assert list(table["HIA"]) == [1.0, 0.0, 0.0, 1.0]
    assert list(table[MEASUREMENT_COUNT_COLUMN]) == [2.0, 1.0, 1.0, 1.0]
    # seed 1 puts train_val row 1 (propanol) into valid, seed 2 row 2 (invalid).
    assert list(table["split__seed1"]) == ["train", "valid", "train", "test"]
    assert list(table["split__seed2"]) == ["train", "train", "train", "test"]
    assert list(table["split"]) == list(table["split__seed1"])
    assert table["enantiomer_of"].isna().tolist() == [True, True, False, False]
    assert table["enantiomer_of"].iloc[2:].tolist() == [3, 2]
    assert bundle.provenance.notices == [AGGREGATION_NOTICE]
    assert bundle.provenance.preparer == PREPARER
    assert set(bundle.provenance.source.files) == set(
        source_file_hashes(raw_root, "hia_hou")
    )
    assert set(bundle.provenance.source.package_versions) == {"PyTDC", "rdkit"}

    summary = format_report([report])
    assert "HIA_Hou" in summary
    assert report.content_sha256[:14] in summary


def test_prepare_endpoint_refuses_a_classification_label_outside_zero_one(
    tmp_path: Path,
) -> None:
    """Caught before the vote, which would otherwise turn 2.0 into class 0."""
    train_val = _frame([("a", ETHANOL, 2.0), ("b", PROPANOL, 0.0)])
    raw_root = tmp_path / "raw"
    _write_raw_endpoint(raw_root, "hia_hou", train_val, _frame([("c", BUTANOL, 1.0)]))
    config = TdcPreparerConfig(
        raw_root=raw_root, bundle_root=tmp_path / "bundles", seeds=[1]
    )
    with pytest.raises(LabelAggregationError, match="neither 0 nor 1"):
        prepare_endpoint(
            endpoint=_endpoint("HIA_Hou"),
            config=config,
            group=FakeAdmetGroup("hia_hou", train_val),
            tdc_metric="roc-auc",
            preparer=PREPARER,
        )
    assert not (tmp_path / "bundles" / "HIA_Hou").exists()


def test_prepare_endpoint_surfaces_a_format_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever slips past the preparer's own checks, ``write_bundle`` refuses."""
    import remedi.data_handling.bundle.aggregation as aggregation_module

    train_val = _frame([("a", ETHANOL, 1.0), ("b", PROPANOL, 0.0)])
    raw_root = tmp_path / "raw"
    _write_raw_endpoint(raw_root, "hia_hou", train_val, _frame([("c", BUTANOL, 1.0)]))
    monkeypatch.setattr(aggregation_module, "aggregate_label", lambda values, _: 3.0)
    config = TdcPreparerConfig(
        raw_root=raw_root, bundle_root=tmp_path / "bundles", seeds=[1]
    )
    with pytest.raises(BundleValidationError, match="outside 0 or 1"):
        prepare_endpoint(
            endpoint=_endpoint("HIA_Hou"),
            config=config,
            group=FakeAdmetGroup("hia_hou", train_val),
            tdc_metric="roc-auc",
            preparer=PREPARER,
        )
    assert not (tmp_path / "bundles" / "HIA_Hou" / "table.parquet").exists()


def test_read_source_table_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TdcPreparationError, match="missing raw file"):
        read_source_table(tmp_path, "hia_hou")


def test_load_admet_group_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(TdcPreparationError, match="never downloads"):
        load_admet_group(tmp_path)


def test_main_returns_one_without_the_download(tmp_path: Path) -> None:
    exit_code = main(
        ["--raw-root", str(tmp_path), "--bundle-root", str(tmp_path / "bundles")]
    )
    assert exit_code == 1
    assert not (tmp_path / "bundles").exists()


# --------------------------------------------------- end to end, real data


@requires_raw_download
def test_prepare_the_smallest_real_endpoint(tmp_path: Path) -> None:
    config = TdcPreparerConfig(
        raw_root=RAW_ROOT,
        bundle_root=tmp_path / "bundles",
        seeds=[1, 2],
        only=["Bioavailability_Ma"],
    )
    report = prepare_endpoint(
        endpoint=_endpoint("Bioavailability_Ma"),
        config=config,
        group=load_admet_group(config.raw_root),
        tdc_metric=tdc_default_metrics()["bioavailability_ma"],
        preparer=PreparerRecord.for_script("molsuit/Rem3Di", Path(__file__)),
    )

    source = read_source_table(RAW_ROOT, "bioavailability_ma")
    counts = report.counts
    assert counts.source_molecules == len(source)
    assert counts.final_rows + sum(counts.dropped.values()) == counts.source_molecules
    assert set(counts.per_split) == {"train", "valid", "test"}
    assert counts.per_split["test"] > 0

    bundle = read_bundle(report.directory)
    assert sorted(path.name for path in report.directory.iterdir()) == [
        "dataset.yaml",
        "provenance.yaml",
        "table.parquet",
    ]
    assert bundle.spec.split_columns() == ["split", "split__seed1", "split__seed2"]
    assert bundle.spec.evaluation is not None
    assert bundle.spec.evaluation.metrics[0] is EvalMetric.auroc
    assert (bundle.table["split"] == bundle.table["split__seed1"]).all()
    assert set(bundle.table["split"]) <= {"train", "valid", "test"}
    assert set(bundle.table["Bioavailability"].unique()) <= {0.0, 1.0}
    assert bundle.provenance.preparer.script == "preparers/tdc/tests/test_prepare.py"
    assert bundle.provenance.smiles_filter is not None
    assert bundle.provenance.smiles_filter.max_atoms == 100


@requires_raw_download
def test_replicates_are_aggregated_on_a_real_endpoint(tmp_path: Path) -> None:
    """PPBR_AZ is the endpoint with the most replicate measurements."""
    config = TdcPreparerConfig(
        raw_root=RAW_ROOT,
        bundle_root=tmp_path / "bundles",
        seeds=[1],
        only=["PPBR_AZ"],
    )
    report = prepare_endpoint(
        endpoint=_endpoint("PPBR_AZ"),
        config=config,
        group=load_admet_group(config.raw_root),
        tdc_metric=tdc_default_metrics()["ppbr_az"],
        preparer=PREPARER,
    )
    dropped = report.counts.dropped
    assert dropped[DUPLICATE_SMILES] > 0
    assert dropped[DUPLICATE_LABEL_TIE] == 0  # a regression endpoint never ties
    assert report.rows_with_replicates > 0

    counts = read_bundle(report.directory).table[MEASUREMENT_COUNT_COLUMN]
    # every source row that survived the filter went into exactly one label
    assert counts.sum() == report.counts.final_rows + dropped[DUPLICATE_SMILES]
    assert counts.max() > 1.0


def _patch_tdc(
    monkeypatch: pytest.MonkeyPatch, group: FakeAdmetGroup, metrics: dict[str, str]
) -> None:
    import remedi_prepare_tdc.prepare as prepare_module

    monkeypatch.setattr(prepare_module, "load_admet_group", lambda raw_root: group)
    monkeypatch.setattr(prepare_module, "tdc_default_metrics", lambda: metrics)


def test_main_prepares_the_selection_and_prints_the_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_val = _frame([("a", ETHANOL, 1.0), ("b", PROPANOL, 0.0)])
    raw_root = tmp_path / "raw"
    _write_raw_endpoint(raw_root, "hia_hou", train_val, _frame([("c", BUTANOL, 1.0)]))
    _patch_tdc(
        monkeypatch, FakeAdmetGroup("hia_hou", train_val), {"hia_hou": "roc-auc"}
    )

    exit_code = main(
        [
            "--raw-root",
            str(raw_root),
            "--bundle-root",
            str(tmp_path / "bundles"),
            "--only",
            "HIA_Hou",
            "--seeds",
            "1",
        ]
    )

    assert exit_code == 0
    assert "HIA_Hou" in capsys.readouterr().out
    bundle = read_bundle(tmp_path / "bundles" / "HIA_Hou")
    assert bundle.provenance.preparer.script == (
        "preparers/tdc/remedi_prepare_tdc/prepare.py"
    )
    assert bundle.spec.split_columns() == ["split", "split__seed1"]


def test_main_fails_on_an_endpoint_without_a_tdc_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_val = _frame([("a", ETHANOL, 1.0)])
    _patch_tdc(monkeypatch, FakeAdmetGroup("hia_hou", train_val), {})
    exit_code = main(["--bundle-root", str(tmp_path / "bundles"), "--only", "HIA_Hou"])
    assert exit_code == 1
    assert not (tmp_path / "bundles").exists()
