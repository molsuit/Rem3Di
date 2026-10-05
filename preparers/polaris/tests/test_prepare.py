"""Tests for the Polaris-specific parts of the Polaris preparer.

The shared mechanics (merge, table, spec, report, CLI) are tested with
``remedi.data_handling.bundle``. Here: the catalog, the dumped source table, the
test-fold rules and the enhanced-stereo step on tiny synthetic inputs,
``prepare_dataset`` end to end on synthetic dumps in ``tmp_path``, and two tests
on the real dumps, skipped when ``benchmark_data/raw/polaris`` is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from remedi_prepare_polaris.prepare import (
    OR_STEREO_GROUP,
    POLARIS_DATASETS,
    PolarisDataset,
    PolarisDatasetCatalog,
    PolarisPreparationError,
    PolarisPreparerConfig,
    assign_test_fold,
    main,
    prepare_dataset,
    read_source_table,
    unspecify_stereo_groups,
)

from remedi.data_handling.bundle import (
    AGGREGATION_NOTICE,
    INVALID_SMILES,
    MEASUREMENT_COUNT_COLUMN,
    SMILES_FILTER,
    EvalMetric,
    IdentityLabelTransform,
    Log10LabelTransform,
    MergedRows,
    PreparerRecord,
    merge_source_rows,
    read_bundle,
)
from remedi.data_handling.bundle.preparation import BundleReport
from remedi.data_handling.chemistry.smiles_filter import SmilesFilterConfig
from remedi.data_handling.dataset.tasks import Split, TaskType

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "polaris"
PREPARER = PreparerRecord(repo="molsuit/Rem3Di", script="prepare.py")

requires_raw_dump = pytest.mark.skipif(
    not (RAW_ROOT / "polaris_antiviral_admet.parquet").is_file(),
    reason="the Polaris parquet dumps are not present",
)

TRAIN, TEST, UNASSIGNED = Split.train.value, Split.test.value, Split.unassigned.value
#: Twelve small molecules on distinct scaffolds, so scaffold splits have room.
SCAFFOLD_MOLECULES = [
    "OCCc1ccccc1",
    "CCc1cccnc1",
    "NC1CCCCC1",
    "c1ccc2ccccc2c1",
    "Cc1ccco1",
    "Cc1cccs1",
    "OCC1CCCO1",
    "c1cnccn1",
    "CC1CCCNC1",
    "O=C1CCCN1",
    "c1cc[nH]c1",
    "c1ccc(-c2ccccc2)cc1",
]
#: The two enantiomers of 1-phenylethylamine, each tagged as an OR group: the
#: sample is one configuration, which one is unknown.
OR_GROUP_R = "C[C@H](N)c1ccccc1 |o1:1|"
OR_GROUP_S = "C[C@@H](N)c1ccccc1 |o1:1|"
#: A racemate: the AND group makes the drawn centre arbitrary.
AND_GROUP_RACEMATE = "C[C@H](N)c1ccccc1 |&1:1|"
PHENYLETHYLAMINE = "CC(N)c1ccccc1"
ADMET_COLUMNS = ["HLM", "KSOL", "LogD", "MDR1-MDCKII", "MLM"]
KINASES = ["EGFR", "KIT", "RET", "LOK", "SLK"]


def _dataset(dataset_id: str) -> PolarisDataset:
    return next(item for item in POLARIS_DATASETS if item.dataset_id == dataset_id)


def _config(tmp_path: Path) -> PolarisPreparerConfig:
    return PolarisPreparerConfig(
        raw_root=tmp_path / "raw", bundle_root=tmp_path / "bundles"
    )


def _write_dump(
    raw_root: Path, dataset: PolarisDataset, frame: pd.DataFrame, slug: str = ""
) -> None:
    """Write a parquet and four-field source yaml the way ``dump_polaris.py`` does."""
    raw_root.mkdir(parents=True, exist_ok=True)
    frame.astype({"split": "uint8"}).to_parquet(
        raw_root / dataset.source_file, index=False
    )
    record = {
        "slug": slug or dataset.slug,
        "polaris_lib_version": "0.13.0",
        "checksum": None,
        "dumped_at": "2026-10-05T15:35:06+00:00",
    }
    (raw_root / dataset.source_record_file).write_text(yaml.safe_dump(record))


def _admet_frame() -> pd.DataFrame:
    """Twelve train/test rows, a racemate (train) and the same compound drawn
    without stereo (test), and one OR-group enantiomer the preparer drops."""
    rows: list[tuple[object, ...]] = [
        (smiles, TRAIN if position < 9 else TEST, position, 100.0, 1.5, 2.0, 4.0)
        for position, smiles in enumerate(SCAFFOLD_MOLECULES)
    ]
    rows.append((AND_GROUP_RACEMATE, TRAIN, 10.0, math.nan, 1.0, 3.0, 5.0))
    rows.append((PHENYLETHYLAMINE, TEST, 30.0, math.nan, 2.0, math.nan, 7.0))
    rows.append((OR_GROUP_S, TRAIN, 99.0, 99.0, 9.0, 9.0, 9.0))
    return pd.DataFrame(rows, columns=["smiles", "split", *ADMET_COLUMNS])


def _pkis2_frame() -> pd.DataFrame:
    rows = [
        (smiles, UNASSIGNED, *([90.0 * (position % 2)] * 5), *([position % 2] * 5))
        for position, smiles in enumerate(SCAFFOLD_MOLECULES)
    ]
    columns = ["smiles", "split", *KINASES, *(f"CLS_{name}" for name in KINASES)]
    return pd.DataFrame(rows, columns=columns).astype(
        {f"CLS_{name}": "float64" for name in KINASES}
    )


def _merge(smiles: list[str]) -> MergedRows:
    return merge_source_rows(
        smiles,
        {"y": [1.0] * len(smiles)},
        {"y": TaskType.regression},
        SmilesFilterConfig(dedupe=False),
    )


# ------------------------------------------------------------- the catalog


def test_the_catalog_lists_five_bundles_and_their_test_fold_kinds() -> None:
    kinds = {dataset.dataset_id: dataset.test_fold.kind for dataset in POLARIS_DATASETS}
    assert kinds == {
        "polaris_antiviral_admet": "shipped",
        "polaris_antiviral_potency": "shipped",
        "polaris_adme_fang": "deepchem_scaffold",
        "polaris_pkis2_subset": "deepchem_scaffold",
        "polaris_pkis2_subset_cls": "deepchem_scaffold",
    }
    assert (
        _dataset("polaris_pkis2_subset").source_file
        == _dataset("polaris_pkis2_subset_cls").source_file
    )
    assert _dataset("polaris_adme_fang").source_record_file == (
        "polaris_adme_fang.source.yaml"
    )


def test_only_the_multilabel_classification_bundle_uses_macro_auroc() -> None:
    for dataset in POLARIS_DATASETS:
        if dataset.task_type is TaskType.classification:
            assert dataset.metrics() == [EvalMetric.macro_auroc]
        else:
            assert dataset.metrics()[0] is EvalMetric.mae
    assert _dataset("polaris_pkis2_subset_cls").task_type is TaskType.classification


def _entry(**overrides: object) -> dict[str, object]:
    return {
        "dataset_id": "A",
        "source_file": "a.parquet",
        "slug": "owner/a",
        "property_description": "x",
        "task_type": "regression",
        "test_fold": {"kind": "shipped"},
        "labels": [{"name": "y"}],
        **overrides,
    }


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        ([_entry(), _entry()], "duplicate dataset_id"),
        ([_entry(labels=[{"name": "y"}, {"name": "y"}])], "duplicate label names"),
        (
            [
                _entry(
                    task_type="classification",
                    labels=[{"name": "y", "transform": {"kind": "log10"}}],
                )
            ],
            "only allowed on regression",
        ),
        (
            [_entry(test_fold={"kind": "deepchem_scaffold", "train_fraction": 0.9})],
            "leaves no test fold",
        ),
    ],
    ids=["duplicate-id", "duplicate-label", "classification-transform", "no-test"],
)
def test_the_catalog_rejects(datasets: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PolarisDatasetCatalog.model_validate({"datasets": datasets})


# -------------------------------------------------------- the source table


ADMET = _dataset("polaris_antiviral_admet")


@pytest.mark.parametrize(
    ("changes", "slug", "record", "message"),
    [
        ({}, "", "slug: x\n", "is malformed"),
        ({}, "someone/else", None, "was dumped from"),
        ({"split": 7}, "", None, r"unknown split codes \[7\]"),
        ({"LogD": None}, "", None, "'LogD' is missing"),
        ({"LogD": "high"}, "", None, "'LogD' is missing"),
    ],
    ids=["malformed record", "foreign slug", "split code", "no label", "text label"],
)
def test_read_source_table_rejects_a_bad_dump(
    tmp_path: Path,
    changes: dict[str, object],
    slug: str,
    record: str | None,
    message: str,
) -> None:
    frame = _admet_frame()
    for column, value in changes.items():
        if value is None:
            frame = frame.drop(columns=column)
        else:
            frame[column] = value
    _write_dump(tmp_path, ADMET, frame, slug=slug)
    if record is not None:
        (tmp_path / ADMET.source_record_file).write_text(record)
    with pytest.raises(PolarisPreparationError, match=message):
        read_source_table(tmp_path, ADMET)


# ------------------------------------------------------------ the test fold


@pytest.mark.parametrize(
    ("dataset_id", "codes", "message"),
    [
        ("polaris_adme_fang", [TRAIN, TEST], "declare 'shipped'"),
        ("polaris_antiviral_admet", [UNASSIGNED] * 2, "declare 'deepchem_scaffold'"),
        ("polaris_antiviral_admet", [TRAIN, UNASSIGNED], "1 of 2 source rows have no"),
    ],
    ids=["codes-for-scaffold", "no-codes-for-shipped", "partial-codes"],
)
def test_the_test_fold_kind_must_match_the_shipped_codes(
    dataset_id: str, codes: list[int], message: str
) -> None:
    merged = _merge(SCAFFOLD_MOLECULES[:2])
    with pytest.raises(PolarisPreparationError, match=message):
        assign_test_fold(_dataset(dataset_id), merged, np.array(codes))


def test_a_bundle_row_is_test_if_any_merged_source_row_was_test() -> None:
    # Rows 0 and 3 merge (train + test), rows 2 and 4 merge (test + test).
    merged = _merge(["CCO", "CCN", "CCC", "OCC", "CCC"])
    assert merged.member_source_rows == [[0, 3], [1], [2, 4]]
    assignment = assign_test_fold(
        _dataset("polaris_antiviral_admet"),
        merged,
        np.array([TRAIN, TRAIN, TEST, TEST, TEST]),
    )
    assert assignment.is_test.tolist() == [True, False, True]
    assert assignment.straddling_rows == 1
    assert "1 bundle rows merge" in assignment.description


# --------------------------------------------------------- enhanced stereo


def test_or_group_enantiomers_collapse_to_one_unspecified_smiles() -> None:
    outcome = unspecify_stereo_groups([OR_GROUP_R, OR_GROUP_S, "not smiles"])
    assert outcome.smiles[0] == outcome.smiles[1] == PHENYLETHYLAMINE
    # An unparseable SMILES passes on unchanged for the filter to count.
    assert outcome.smiles[2] == "not smiles"
    assert (outcome.rows_changed, outcome.centres_unspecified) == (2, 2)
    assert "2 kept source rows had 2 centres" in outcome.notice()


# ------------------------------------------- end to end on synthetic dumps


@pytest.fixture
def admet_report(tmp_path: Path) -> BundleReport:
    dataset = _dataset("polaris_antiviral_admet")
    config = _config(tmp_path)
    _write_dump(config.raw_root, dataset, _admet_frame())
    return prepare_dataset(dataset, config, PREPARER)


def test_prepare_dataset_writes_a_shipped_split_bundle(
    admet_report: BundleReport,
) -> None:
    bundle = read_bundle(admet_report.directory)
    table = bundle.table
    # Twelve molecules plus the racemate merged with its stereo-free drawing;
    # the OR-group enantiomer is dropped and counted.
    counts = admet_report.counts
    assert (counts.source_molecules, counts.final_rows) == (15, 13)
    assert counts.dropped[OR_STEREO_GROUP] == 1
    assert admet_report.extra == {
        "stereo_unspecified_rows": 1,
        "test_straddling_rows": 1,
    }
    merged = table[table["isomeric_smiles"] == PHENYLETHYLAMINE].iloc[0]
    assert merged["HLM"] == pytest.approx(20.0)
    assert math.isnan(merged["KSOL"])
    assert merged[MEASUREMENT_COUNT_COLUMN] == 2.0
    # The merged row had a test member, so it is test in every split column.
    for column_name in bundle.spec.split_columns():
        assert merged[column_name] == "test"
        assert int((table[column_name] == "test").sum()) == 4

    assert bundle.spec.source_kind == "polaris_hub"
    provenance = bundle.provenance
    assert set(provenance.source.files) == {
        "polaris_antiviral_admet.parquet",
        "polaris_antiviral_admet.source.yaml",
    }
    assert provenance.source.package_versions["polaris-lib"] == "0.13.0"
    aggregation, stereo, split = provenance.notices
    assert aggregation == AGGREGATION_NOTICE
    assert "1 kept source rows had 1 centres" in stereo
    assert f"counts.dropped.{OR_STEREO_GROUP}" in stereo
    assert "shipped Set column" in split and "1 bundle rows merge" in split


def test_the_log10_transform_is_declared_for_hlm_but_not_logd(
    admet_report: BundleReport,
) -> None:
    written = yaml.safe_load((admet_report.directory / "dataset.yaml").read_text())
    labels = {label["name"]: label for label in written["labels"]}
    assert "transform" not in labels["LogD"]
    for name in ("HLM", "MLM", "KSOL", "MDR1-MDCKII"):
        assert labels[name]["transform"] == {
            "kind": "log10",
            "clip_minimum": 0.0,
            "offset": 1.0,
        }
    transforms = {
        label.name: label.transform
        for label in read_bundle(admet_report.directory).spec.labels
    }
    assert isinstance(transforms["LogD"], IdentityLabelTransform)
    assert transforms["HLM"] == Log10LabelTransform(clip_minimum=0.0, offset=1.0)


def test_main_derives_a_scaffold_test_fold_without_straddling_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_ids = ["polaris_pkis2_subset", "polaris_pkis2_subset_cls"]
    _write_dump(tmp_path / "raw", _dataset(dataset_ids[0]), _pkis2_frame())
    arguments = ["--raw-root", str(tmp_path / "raw")]
    arguments += ["--bundle-root", str(tmp_path / "bundles")]
    for dataset_id in dataset_ids:
        arguments += ["--only", dataset_id]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert "test_straddling_rows" in output
    assert output.splitlines()[-1].split()[-2] == "-"
    for dataset_id in dataset_ids:
        bundle = read_bundle(tmp_path / "bundles" / dataset_id)
        is_test = [
            bundle.table[column] == "test" for column in bundle.spec.split_columns()
        ]
        assert 0 < is_test[0].sum() < len(SCAFFOLD_MOLECULES)
        assert all((column == is_test[0]).all() for column in is_test)
    assert set(bundle.table["CLS_KIT"].unique()) == {0.0, 1.0}
    # The admet dump is absent.
    assert main([*arguments[:4], "--only", "polaris_antiviral_admet"]) == 1


# ---------------------------------------------------------- the real dumps


@requires_raw_dump
def test_the_real_antiviral_admet_dump(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    report = prepare_dataset(
        dataset, PolarisPreparerConfig(bundle_root=tmp_path), PREPARER
    )
    counts = report.counts
    assert counts.source_molecules == 560
    assert counts.dropped[INVALID_SMILES] == 0
    assert counts.dropped[SMILES_FILTER] > 0  # boron-containing compounds
    assert int(report.extra["stereo_unspecified_rows"]) > 0
    assert set(counts.per_split) == {"train", "valid", "test"}


@requires_raw_dump
def test_label_names_with_spaces_and_parentheses_round_trip(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_potency")
    report = prepare_dataset(
        dataset, PolarisPreparerConfig(bundle_root=tmp_path), PREPARER
    )
    bundle = read_bundle(report.directory)
    assert bundle.spec.label_names() == [
        "pIC50 (MERS-CoV Mpro)",
        "pIC50 (SARS-CoV-2 Mpro)",
    ]
    assert [task.name for task in bundle.spec.task_set().system_cols] == (
        bundle.spec.label_names()
    )
    assert "pIC50 (MERS-CoV Mpro)" in bundle.table.columns
