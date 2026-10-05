"""Tests for the Polaris-specific parts of the Polaris preparer.

The shared mechanics (table building, settings, report, CLI parsing, the merge)
are tested with ``remedi.data_handling.bundle.preparation``. Here: the catalog,
the dumped source table, the test-fold rules and the enhanced-stereo step on
tiny synthetic inputs, ``prepare_dataset`` end to end on synthetic dumps in
``tmp_path``, and two tests on the real dumps, skipped when
``benchmark_data/raw/polaris`` is absent.
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
    DeepchemScaffoldTestFold,
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
    ("catalog", "message"),
    [
        ({"datasets": [_entry(), _entry()]}, "duplicate dataset_id"),
        (
            {"datasets": [_entry(labels=[{"name": "y"}, {"name": "y"}])]},
            "duplicate label names",
        ),
        ({"datasets": [_entry(test_fold={"kind": "random"})]}, "random"),
        (
            {
                "datasets": [
                    _entry(
                        task_type="classification",
                        labels=[{"name": "y", "transform": {"kind": "log10"}}],
                    )
                ]
            },
            "only allowed on regression",
        ),
    ],
    ids=["duplicate-id", "duplicate-label", "unknown-fold", "classification-transform"],
)
def test_the_catalog_rejects(catalog: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PolarisDatasetCatalog.model_validate(catalog)


def test_a_label_reads_its_source_column_or_its_own_name() -> None:
    dataset = PolarisDataset.model_validate(
        _entry(labels=[{"name": "y"}, {"name": "z", "source_column": "Z raw"}])
    )
    assert [label.column for label in dataset.labels] == ["y", "Z raw"]
    assert dataset.source_record_file == "a.source.yaml"


def test_only_the_multilabel_classification_bundle_uses_macro_auroc() -> None:
    for dataset in POLARIS_DATASETS:
        if dataset.task_type is TaskType.classification:
            assert dataset.metrics() == [EvalMetric.macro_auroc]
        else:
            assert dataset.metrics()[0] is EvalMetric.mae
            assert EvalMetric.rmse in dataset.metrics()
    assert _dataset("polaris_pkis2_subset_cls").task_type is TaskType.classification


# -------------------------------------------------------- the source table


def test_read_source_table_reads_the_four_field_record(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    _write_dump(tmp_path, dataset, _admet_frame())
    source = read_source_table(tmp_path, dataset)
    assert len(source) == 15
    assert source.record.slug == dataset.slug
    assert source.record.checksum is None
    assert set(source.file_hashes) == {
        "polaris_antiviral_admet.parquet",
        "polaris_antiviral_admet.source.yaml",
    }


def test_read_source_table_rejects_a_missing_dump(tmp_path: Path) -> None:
    with pytest.raises(PolarisPreparationError, match="run preparers/polaris"):
        read_source_table(tmp_path, _dataset("polaris_antiviral_admet"))


def test_read_source_table_rejects_a_slug_mismatch(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    _write_dump(tmp_path, dataset, _admet_frame(), slug="someone/else")
    with pytest.raises(PolarisPreparationError, match="was dumped from"):
        read_source_table(tmp_path, dataset)


def test_read_source_table_rejects_an_unknown_split_code(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    frame = _admet_frame()
    frame.loc[0, "split"] = 7
    _write_dump(tmp_path, dataset, frame)
    with pytest.raises(PolarisPreparationError, match=r"unknown split codes \[7\]"):
        read_source_table(tmp_path, dataset)


@pytest.mark.parametrize("broken", ["missing", "text"])
def test_read_source_table_rejects_a_bad_label_column(
    tmp_path: Path, broken: str
) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    frame = _admet_frame()
    if broken == "missing":
        frame = frame.drop(columns=["LogD"])
    else:
        frame["LogD"] = frame["LogD"].astype(str)
    _write_dump(tmp_path, dataset, frame)
    with pytest.raises(PolarisPreparationError, match="'LogD' is missing"):
        read_source_table(tmp_path, dataset)


# ------------------------------------------------------------ the test fold


def test_shipped_split_codes_need_a_shipped_catalog_entry() -> None:
    merged = _merge(SCAFFOLD_MOLECULES[:2])
    with pytest.raises(PolarisPreparationError, match="declare 'shipped'"):
        assign_test_fold(_dataset("polaris_adme_fang"), merged, np.array([TRAIN, TEST]))


@pytest.mark.parametrize(
    ("codes", "message"),
    [
        ([UNASSIGNED, UNASSIGNED], "declare 'deepchem_scaffold'"),
        ([TRAIN, UNASSIGNED], "1 of 2 source rows have no shipped split"),
    ],
    ids=["none", "partial"],
)
def test_a_shipped_catalog_entry_needs_every_row_assigned(
    codes: list[int], message: str
) -> None:
    merged = _merge(SCAFFOLD_MOLECULES[:2])
    with pytest.raises(PolarisPreparationError, match=message):
        assign_test_fold(_dataset("polaris_antiviral_admet"), merged, np.array(codes))


def test_a_bundle_row_is_test_if_any_merged_source_row_was_test() -> None:
    # Rows 0 and 3 merge (train + test), rows 2 and 4 merge (test + test).
    smiles = ["CCO", "CCN", "CCC", "OCC", "CCC"]
    merged = _merge(smiles)
    assert merged.member_source_rows == [[0, 3], [1], [2, 4]]
    assignment = assign_test_fold(
        _dataset("polaris_antiviral_admet"),
        merged,
        np.array([TRAIN, TRAIN, TEST, TEST, TEST]),
    )
    assert assignment.is_test.tolist() == [True, False, True]
    assert assignment.straddling_rows == 1
    assert "1 bundle rows merge" in assignment.description


def test_a_scaffold_test_fold_is_derived_without_straddling_count() -> None:
    merged = _merge(SCAFFOLD_MOLECULES)
    assignment = assign_test_fold(
        _dataset("polaris_adme_fang"),
        merged,
        np.full(len(SCAFFOLD_MOLECULES), UNASSIGNED),
    )
    assert 0 < assignment.is_test.sum() < len(SCAFFOLD_MOLECULES)
    assert assignment.straddling_rows is None
    assert isinstance(_dataset("polaris_adme_fang").test_fold, DeepchemScaffoldTestFold)


# --------------------------------------------------------- enhanced stereo


def test_or_group_enantiomers_collapse_to_one_unspecified_smiles() -> None:
    outcome = unspecify_stereo_groups(
        [OR_GROUP_R, OR_GROUP_S, "C[C@H](N)c1ccccc1 |a:1|", "not smiles"]
    )
    assert outcome.smiles[0] == outcome.smiles[1]
    assert "@" not in outcome.smiles[0]
    # Absolute centres keep their configuration.
    assert "@" in outcome.smiles[2]
    # An unparseable SMILES passes on unchanged for the filter to count.
    assert outcome.smiles[3] == "not smiles"
    assert (outcome.rows_changed, outcome.centres_unspecified) == (2, 2)
    assert "2 kept source rows had 2 centres" in outcome.notice()


# ------------------------------------------- end to end on synthetic dumps


def test_prepare_dataset_writes_a_shipped_split_bundle(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    config = _config(tmp_path)
    _write_dump(config.raw_root, dataset, _admet_frame())

    report = prepare_dataset(dataset, config, PREPARER)
    bundle = read_bundle(report.directory)
    table = bundle.table

    # Twelve molecules plus the racemate merged with its stereo-free drawing;
    # the OR-group enantiomer is dropped and counted.
    assert (report.counts.source_molecules, report.counts.final_rows) == (15, 13)
    assert report.counts.dropped[OR_STEREO_GROUP] == 1
    assert report.extra == {"stereo_unspecified_rows": 1, "test_straddling_rows": 1}
    merged = table[table["isomeric_smiles"] == PHENYLETHYLAMINE]
    assert len(merged) == 1
    assert merged["HLM"].iloc[0] == pytest.approx(20.0)
    assert merged["LogD"].iloc[0] == pytest.approx(1.5)
    assert merged["MDR1-MDCKII"].iloc[0] == pytest.approx(3.0)
    assert math.isnan(merged["KSOL"].iloc[0])
    assert merged["MLM"].iloc[0] == pytest.approx(6.0)
    assert merged[MEASUREMENT_COUNT_COLUMN].iloc[0] == 2.0
    # The merged row had a test member, so it is test in every split column.
    for column_name in config.split_columns():
        assert merged[column_name].iloc[0] == "test"
        assert int((table[column_name] == "test").sum()) == 4

    assert bundle.spec.source_kind == "polaris_hub"
    assert bundle.spec.label_names() == ["LogD", "HLM", "MLM", "KSOL", "MDR1-MDCKII"]
    provenance = bundle.provenance
    assert set(provenance.source.files) == {
        "polaris_antiviral_admet.parquet",
        "polaris_antiviral_admet.source.yaml",
    }
    assert provenance.source.package_versions["polaris-lib"] == "0.13.0"
    assert set(provenance.source.package_versions) == {"polaris-lib", "rdkit"}
    assert provenance.smiles_filter is not None
    assert provenance.smiles_filter.dedupe is False
    aggregation, stereo, split = provenance.notices
    assert aggregation == AGGREGATION_NOTICE
    assert "1 kept source rows had 1 centres" in stereo
    assert f"counts.dropped.{OR_STEREO_GROUP}" in stereo
    assert "shipped Set column" in split and "1 bundle rows merge" in split


def test_the_log10_transform_is_declared_for_hlm_but_not_logd(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    config = _config(tmp_path)
    _write_dump(config.raw_root, dataset, _admet_frame())
    report = prepare_dataset(dataset, config, PREPARER)

    written = yaml.safe_load((report.directory / "dataset.yaml").read_text())
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
        for label in read_bundle(report.directory).spec.labels
    }
    assert isinstance(transforms["LogD"], IdentityLabelTransform)
    assert transforms["HLM"] == Log10LabelTransform(clip_minimum=0.0, offset=1.0)


def test_prepare_dataset_derives_a_scaffold_test_fold(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for dataset_id in ("polaris_pkis2_subset", "polaris_pkis2_subset_cls"):
        dataset = _dataset(dataset_id)
        _write_dump(config.raw_root, dataset, _pkis2_frame())
        report = prepare_dataset(dataset, config, PREPARER)
        bundle = read_bundle(report.directory)
        table = bundle.table
        assert report.extra["test_straddling_rows"] == "-"
        assert report.counts.final_rows == len(SCAFFOLD_MOLECULES)
        test_rows = set(table.index[table["split"] == "test"])
        assert test_rows
        for column_name in config.split_columns():
            assert set(table.index[table[column_name] == "test"]) == test_rows
        assert bundle.spec.evaluation is not None
        assert bundle.spec.evaluation.metrics == dataset.metrics()
    classification = read_bundle(tmp_path / "bundles" / "polaris_pkis2_subset_cls")
    assert set(classification.table["CLS_KIT"].unique()) == {0.0, 1.0}


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = _dataset("polaris_pkis2_subset")
    _write_dump(tmp_path / "raw", dataset, _pkis2_frame())
    roots = ["--raw-root", str(tmp_path / "raw")]
    roots += ["--bundle-root", str(tmp_path / "bundles")]
    assert main([*roots, "--only", "polaris_pkis2_subset"]) == 0
    assert "polaris_pkis2_subset" in capsys.readouterr().out
    assert (tmp_path / "bundles" / "polaris_pkis2_subset" / "table.parquet").is_file()
    # The admet dump is absent.
    assert main([*roots, "--only", "polaris_antiviral_admet"]) == 1


# ---------------------------------------------------------- the real dumps


@requires_raw_dump
def test_the_real_antiviral_admet_dump(tmp_path: Path) -> None:
    dataset = _dataset("polaris_antiviral_admet")
    report = prepare_dataset(
        dataset, PolarisPreparerConfig(bundle_root=tmp_path), PREPARER
    )
    counts = report.counts
    assert counts.source_molecules == 560
    assert counts.final_rows + sum(counts.dropped.values()) == 560
    assert counts.dropped[INVALID_SMILES] == 0
    assert counts.dropped[SMILES_FILTER] > 0  # boron-containing compounds
    assert int(report.extra["stereo_unspecified_rows"]) > 0
    assert set(counts.per_split) == {"train", "valid", "test"}
    bundle = read_bundle(report.directory)
    assert bundle.spec.evaluation is not None
    assert bundle.spec.evaluation.metrics[0] is EvalMetric.mae


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
