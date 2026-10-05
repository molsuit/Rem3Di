"""Tests for the TDC-specific parts of the TDC ADMET preparer.

The shared mechanics (merge, table, spec, report, CLI) are tested with
``remedi.data_handling.bundle``. Here: the endpoint catalog, the metric mapping,
the PyTDC quirks, mapping PyTDC's train/valid partitions back onto the source
rows, ``prepare_endpoint`` end to end on synthetic csvs with a stand-in for the
PyTDC benchmark group, and two tests on real endpoints, skipped when the pinned
``admet_group`` download is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from remedi_prepare_tdc.prepare import (
    ENDPOINTS_FILE,
    TDC_ENDPOINTS,
    TdcEndpoint,
    TdcEndpointCatalog,
    TdcPreparationError,
    TdcPreparerConfig,
    build_spec,
    headline_metric,
    load_admet_group,
    main,
    patch_tdc_print_sys,
    prepare_endpoint,
    read_source_table,
    run,
    source_file_hashes,
    task_type_for_metric,
    tdc_default_metrics,
    train_valid_labels,
)

from remedi.data_handling.bundle import (
    AGGREGATION_NOTICE,
    DUPLICATE_LABEL_TIE,
    DUPLICATE_SMILES,
    MEASUREMENT_COUNT_COLUMN,
    EvalMetric,
    PreparerRecord,
    read_bundle,
)
from remedi.data_handling.bundle.preparation import BundleReport
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


def test_the_catalog_lists_22_endpoints_in_run_order() -> None:
    dataset_ids = [endpoint.dataset_id for endpoint in TDC_ENDPOINTS]
    assert len(set(dataset_ids)) == 22
    assert len({endpoint.label_name for endpoint in TDC_ENDPOINTS}) == 22
    assert TdcEndpointCatalog.from_yaml(ENDPOINTS_FILE).dataset_ids() == dataset_ids
    selected = TdcPreparerConfig(only=["DILI", "HIA_Hou"]).selected_endpoints()
    assert [endpoint.dataset_id for endpoint in selected] == ["HIA_Hou", "DILI"]


@pytest.mark.parametrize(
    ("second", "message"),
    [
        (("A", "b"), "duplicate dataset_id"),
        (("B", "a"), "duplicate label_name"),
    ],
)
def test_the_catalog_rejects_a_duplicate_name(
    second: tuple[str, str], message: str
) -> None:
    endpoints = [
        {"dataset_id": dataset_id, "label_name": label, "property_description": "x"}
        for dataset_id, label in [("A", "a"), second]
    ]
    with pytest.raises(ValueError, match=message):
        TdcEndpointCatalog.model_validate({"endpoints": endpoints})


# ------------------------------------------------------------------ metrics


@pytest.mark.parametrize(
    ("tdc_metric", "metric", "task_type"),
    [
        ("mae", EvalMetric.mae, TaskType.regression),
        ("spearman", EvalMetric.spearman, TaskType.regression),
        ("roc-auc", EvalMetric.auroc, TaskType.classification),
        ("pr-auc", EvalMetric.auprc, TaskType.classification),
    ],
)
def test_every_tdc_metric_maps_to_a_metric_and_task_type(
    tdc_metric: str, metric: EvalMetric, task_type: TaskType
) -> None:
    assert headline_metric(tdc_metric) is metric
    assert task_type_for_metric(metric) is task_type


def test_an_unmapped_metric_is_rejected() -> None:
    with pytest.raises(TdcPreparationError, match="no EvalMetric equivalent"):
        headline_metric("accuracy")
    with pytest.raises(TdcPreparationError, match="belongs to no task type"):
        task_type_for_metric(EvalMetric.macro_f1)


@pytest.mark.parametrize(
    ("dataset_id", "tdc_metric", "label", "metrics"),
    [
        (
            "Caco2_Wang",
            "mae",
            "Caco-2",
            [EvalMetric.mae, EvalMetric.rmse, EvalMetric.spearman, EvalMetric.r2],
        ),
        ("DILI", "pr-auc", "DILI", [EvalMetric.auprc, EvalMetric.auroc]),
    ],
)
def test_build_spec_puts_the_tdc_headline_first(
    dataset_id: str, tdc_metric: str, label: str, metrics: list[EvalMetric]
) -> None:
    spec = build_spec(_endpoint(dataset_id), tdc_metric, ["split"])
    assert spec.source_kind == "tdc_admet_group"
    assert dataset_id in spec.description
    assert [column.name for column in spec.labels] == [label]
    assert spec.evaluation is not None
    assert spec.evaluation.metrics == metrics


# ------------------------------------------------------------- PyTDC quirks


def test_print_sys_is_injected_into_the_pytdc_split_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyTDC's scaffold split logs skipped SMILES via a ``print_sys`` it never imports."""
    import tdc.utils.split as tdc_split_module

    monkeypatch.delattr(tdc_split_module, "print_sys", raising=False)
    patch_tdc_print_sys()
    assert hasattr(tdc_split_module, "print_sys")
    patch_tdc_print_sys()  # idempotent


def test_the_source_is_never_downloaded(tmp_path: Path) -> None:
    with pytest.raises(TdcPreparationError, match="never downloads"):
        load_admet_group(tmp_path)
    with pytest.raises(TdcPreparationError, match="missing raw file"):
        read_source_table(tmp_path, "hia_hou")


# -------------------------------------------------------- the split columns


@pytest.mark.parametrize(
    ("train_val", "train", "valid"),
    [
        (
            [("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0), ("c", BUTANOL, 0.0)],
            [("c", BUTANOL, 0.0), ("a", ETHANOL, 0.0)],
            [("b", PROPANOL, 1.0)],
        ),
        # Identical rows are interchangeable; the earlier position keeps train.
        (
            [("a", ETHANOL, 0.0), ("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)],
            [("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)],
            [("a", ETHANOL, 0.0)],
        ),
    ],
    ids=["reordered copies", "identical rows"],
)
def test_train_valid_labels_matches_the_copies_back_by_content(
    train_val: list, train: list, valid: list
) -> None:
    labels = train_valid_labels(_frame(train_val), _frame(train), _frame(valid))
    assert labels == ["train", "valid", "train"]


@pytest.mark.parametrize(
    ("train", "message"),
    [
        ([("a", ETHANOL, 0.0)], "neither the train nor the valid"),
        ([("z", BUTANOL, 1.0)], "not an unclaimed train_val row"),
    ],
    ids=["dropped row", "foreign row"],
)
def test_train_valid_labels_rejects_a_partition_that_does_not_match(
    train: list, message: str
) -> None:
    train_val = _frame([("a", ETHANOL, 0.0), ("b", PROPANOL, 1.0)])
    with pytest.raises(TdcPreparationError, match=message):
        train_valid_labels(train_val, _frame(train), _frame([]))


class FakeAdmetGroup:
    """Stands in for PyTDC's ``admet_group``: one endpoint, one partition per seed.

    Seed ``s`` puts train_val row ``s mod n`` into valid and the rest into train,
    returned as re-indexed copies the way PyTDC does.
    """

    def __init__(self, directory_name: str, train_val: pd.DataFrame):
        self.dataset_names = [directory_name]
        self.train_val = train_val

    def get_train_valid_split(
        self, seed: int, benchmark: str, split_type: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert split_type == "default"
        is_valid = pd.RangeIndex(len(self.train_val)) == seed % len(self.train_val)
        train = self.train_val[~is_valid].reset_index(drop=True)
        valid = self.train_val[is_valid].reset_index(drop=True)
        return train, valid


# ------------------------------------------- end to end on synthetic csvs


def _write_raw_endpoint(
    raw_root: Path, train_val: pd.DataFrame, test: pd.DataFrame
) -> None:
    directory = raw_root / "admet_group" / "hia_hou"
    directory.mkdir(parents=True)
    train_val.to_csv(directory / "train_val.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)


def _prepare_hia_hou(
    tmp_path: Path, train_val: pd.DataFrame, test: pd.DataFrame
) -> BundleReport:
    raw_root = tmp_path / "raw"
    _write_raw_endpoint(raw_root, train_val, test)
    config = TdcPreparerConfig(
        raw_root=raw_root, bundle_root=tmp_path / "bundles", seeds=[1, 2]
    )
    return prepare_endpoint(
        endpoint=_endpoint("HIA_Hou"),
        config=config,
        group=FakeAdmetGroup("hia_hou", train_val),
        tdc_metric="roc-auc",
        preparer=PREPARER,
    )


def test_prepare_endpoint_takes_pytdc_partitions_and_train_val_priority(
    tmp_path: Path,
) -> None:
    train_val = _frame(
        [
            ("a", ETHANOL, 1.0),
            ("b", PROPANOL, 0.0),
            ("c", "not a molecule", 1.0),
            ("d", BUTANOL, 1.0),
            ("e", L_ALANINE, 0.0),
        ]
    )
    # Ethanol reappears in test: the merged row keeps its train_val split.
    test = _frame([("f", "OCC", 1.0), ("g", D_ALANINE, 1.0)])
    report = _prepare_hia_hou(tmp_path, train_val, test)

    assert (report.counts.source_molecules, report.counts.final_rows) == (7, 5)
    bundle = read_bundle(report.directory)
    table = bundle.table
    assert list(table["isomeric_smiles"]) == [
        ETHANOL,
        PROPANOL,
        BUTANOL,
        L_ALANINE,
        D_ALANINE,
    ]
    assert list(table[MEASUREMENT_COUNT_COLUMN]) == [2.0, 1.0, 1.0, 1.0, 1.0]
    # PyTDC's partitions mapped onto first source rows: seed 1 puts train_val
    # row 1 (propanol) into valid, seed 2 row 2 (the invalid SMILES).
    assert list(table["split__seed1"]) == ["train", "valid", "train", "train", "test"]
    assert list(table["split__seed2"]) == ["train"] * 4 + ["test"]
    assert list(table["split"]) == list(table["split__seed1"])
    assert bundle.spec.labels[0].task_type is TaskType.classification
    provenance = bundle.provenance
    assert provenance.notices == [AGGREGATION_NOTICE]
    assert set(provenance.source.files) == set(
        source_file_hashes(tmp_path / "raw", "hia_hou")
    )
    assert set(provenance.source.package_versions) == {"PyTDC", "rdkit"}


def test_prepare_endpoint_refuses_a_row_without_a_label(tmp_path: Path) -> None:
    train_val = _frame([("a", ETHANOL, 1.0), ("b", PROPANOL, math.nan)])
    with pytest.raises(TdcPreparationError, match="TDC always supplies one"):
        _prepare_hia_hou(tmp_path, train_val, _frame([("c", BUTANOL, 1.0)]))


def _patch_tdc(
    monkeypatch: pytest.MonkeyPatch, group: FakeAdmetGroup, metrics: dict[str, str]
) -> None:
    import remedi_prepare_tdc.prepare as prepare_module

    monkeypatch.setattr(prepare_module, "load_admet_group", lambda raw_root: group)
    monkeypatch.setattr(prepare_module, "tdc_default_metrics", lambda: metrics)


def test_main_prepares_the_selection_with_the_given_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_val = _frame([("a", ETHANOL, 1.0), ("b", PROPANOL, 0.0)])
    _write_raw_endpoint(tmp_path / "raw", train_val, _frame([("c", BUTANOL, 1.0)]))
    _patch_tdc(
        monkeypatch, FakeAdmetGroup("hia_hou", train_val), {"hia_hou": "roc-auc"}
    )
    arguments = ["--raw-root", str(tmp_path / "raw")]
    arguments += ["--bundle-root", str(tmp_path / "bundles")]
    assert main([*arguments, "--only", "HIA_Hou", "--seeds", "1"]) == 0
    assert "HIA_Hou" in capsys.readouterr().out
    bundle = read_bundle(tmp_path / "bundles" / "HIA_Hou")
    assert bundle.provenance.preparer.script == (
        "preparers/tdc/remedi_prepare_tdc/prepare.py"
    )
    assert bundle.spec.split_columns() == ["split", "split__seed1"]


def test_run_fails_on_an_endpoint_without_a_tdc_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tdc(monkeypatch, FakeAdmetGroup("hia_hou", _frame([])), {})
    config = TdcPreparerConfig(bundle_root=tmp_path / "bundles", only=["HIA_Hou"])
    with pytest.raises(
        TdcPreparationError, match=r"not in tdc\.metadata\.admet_metrics"
    ):
        run(config)
    assert not (tmp_path / "bundles").exists()


# --------------------------------------------------- end to end, real data


@requires_raw_download
def test_prepare_the_smallest_real_endpoint(tmp_path: Path) -> None:
    config = TdcPreparerConfig(
        raw_root=RAW_ROOT, bundle_root=tmp_path / "bundles", seeds=[1, 2]
    )
    report = prepare_endpoint(
        endpoint=_endpoint("Bioavailability_Ma"),
        config=config,
        group=load_admet_group(config.raw_root),
        tdc_metric=tdc_default_metrics()["bioavailability_ma"],
        preparer=PREPARER,
    )
    counts = report.counts
    assert counts.source_molecules == len(
        read_source_table(RAW_ROOT, "bioavailability_ma")
    )
    assert set(counts.per_split) == {"train", "valid", "test"}
    bundle = read_bundle(report.directory)
    assert bundle.spec.evaluation is not None
    assert bundle.spec.evaluation.metrics[0] is EvalMetric.auroc
    assert set(bundle.table["Bioavailability"].unique()) <= {0.0, 1.0}


@requires_raw_download
def test_replicates_are_aggregated_on_a_real_endpoint(tmp_path: Path) -> None:
    """PPBR_AZ is the endpoint with the most replicate measurements."""
    config = TdcPreparerConfig(
        raw_root=RAW_ROOT, bundle_root=tmp_path / "bundles", seeds=[1]
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
    counts = read_bundle(report.directory).table[MEASUREMENT_COUNT_COLUMN]
    # every source row that survived the filter went into exactly one label
    assert counts.sum() == report.counts.final_rows + dropped[DUPLICATE_SMILES]
    assert counts.max() > 1.0
