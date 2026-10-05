"""Tests for the MoleculeNet preparer.

Only MoleculeNet-specific behaviour is tested here; the generic mechanics of
:mod:`remedi.data_handling.bundle.preparation` have their own tests. The
catalog, the pin check, the download and the csv reader are exercised on tiny
synthetic files, :func:`run` goes end to end on a synthetic two-dataset catalog
whose pins are computed in the test, and one test runs the smallest real dataset
(freesolv), skipped when the pinned raw file is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from remedi_prepare_moleculenet.prepare import (
    DATASETS_FILE,
    MOLECULENET_CATALOG,
    MoleculeNetCatalog,
    MoleculeNetDataset,
    MoleculeNetPreparationError,
    MoleculeNetPreparerConfig,
    download_missing_file,
    ensure_source_files,
    main,
    prepare_dataset,
    read_source_table,
    run,
    verify_source_file,
)

from remedi.data_handling.bundle import (
    AGGREGATION_NOTICE,
    DUPLICATE_LABEL_TIE,
    DUPLICATE_SMILES,
    INVALID_SMILES,
    MEASUREMENT_COUNT_COLUMN,
    MISSING_LABEL,
    SMILES_FILTER,
    EvalMetric,
    LabelAggregationError,
    PreparerRecord,
    read_bundle,
    sha256_of_file,
)
from remedi.data_handling.chemistry.splits import scaffold_test_mask

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "moleculenet"
PREPARER = PreparerRecord(repo="molsuit/Rem3Di", script="prepare.py")
PIN = "0" * 64
#: The catalog in run order, grouped by how the datasets are scored.
REGRESSION_IDS = ["esol", "freesolv", "lipophilicity", "bace_pic50"]
CLASSIFICATION_IDS = ["bace", "bbbp", "hiv"]
MULTI_LABEL_IDS = ["clintox", "sider", "tox21"]

requires_raw_freesolv = pytest.mark.skipif(
    not (RAW_ROOT / "SAMPL.csv").is_file(),
    reason="the pinned DeepChem S3 SAMPL.csv is not present",
)

#: Twenty molecules with twenty different Bemis-Murcko scaffolds, so the
#: DeepChem scaffold split puts two of them into test.
RING_MOLECULES = """
    Cc1ccccc1 Cc1ccncc1 CC1CCCCC1 CC1CCCC1 CC1CCC1 CC1CC1 Cc1ccoc1 Cc1ccsc1
    Cc1cc[nH]c1 CC1CCNCC1 CC1CCOCC1 Cc1ccc2ccccc2c1 Cc1cnccn1 CC1CCCCCC1
    Cc1ncncn1 CC1CCCCCCC1 Cc1ccc(-c2ccccc2)cc1 CC1CCC2CCCCC2C1
    Cc1ccc2[nH]ccc2c1 Cc1cncnc1
""".split()


def _dataset(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "dataset_id": "toy",
        "source_file": "toy.csv",
        "smiles_column": "smiles",
        "source_rows": 3,
        "task_type": "regression",
        "property_description": "a toy property",
        "labels": [{"name": "toy_label", "source_column": "value"}],
    }
    entry.update(overrides)
    return entry


def _catalog(
    datasets: list[dict[str, Any]],
    source_files: dict[str, str],
    base_url: str = "https://example.invalid/",
) -> MoleculeNetCatalog:
    return MoleculeNetCatalog.model_validate(
        {"base_url": base_url, "source_files": source_files, "datasets": datasets}
    )


def _write_csv(raw_root: Path, file_name: str, frame: pd.DataFrame) -> str:
    """Write ``frame`` (gzip by suffix) and return the file's sha256."""
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / file_name
    frame.to_csv(path, index=False)
    return sha256_of_file(path)


# --------------------------------------------------------------- catalog


def test_the_catalog_lists_ten_datasets_from_nine_files() -> None:
    catalog = MoleculeNetCatalog.from_yaml(DATASETS_FILE)
    assert catalog == MOLECULENET_CATALOG
    by_id = {dataset.dataset_id: dataset for dataset in catalog.datasets}
    assert list(by_id) == [*REGRESSION_IDS, *CLASSIFICATION_IDS, *MULTI_LABEL_IDS]
    assert len(catalog.source_files) == 9
    assert set(catalog.source_files) == {
        dataset.source_file for dataset in catalog.datasets
    }
    assert [label.column for label in by_id["esol"].labels] == [
        "measured log solubility in mols per litre"
    ]
    assert by_id["bace"].smiles_column == "mol"
    assert [label.name for label in by_id["clintox"].labels] == [
        "FDA_APPROVED",
        "CT_TOX",
    ]
    assert len(by_id["sider"].labels) == 27
    assert len(by_id["tox21"].labels) == 12
    for dataset_id in MULTI_LABEL_IDS:
        assert all(label.source_column is None for label in by_id[dataset_id].labels)


def test_the_metrics_follow_the_task_type() -> None:
    regression = [EvalMetric.rmse, EvalMetric.mae, EvalMetric.spearman, EvalMetric.r2]
    classification = [EvalMetric.auroc, EvalMetric.auprc]
    expected = {dataset_id: regression for dataset_id in REGRESSION_IDS}
    expected |= {dataset_id: classification for dataset_id in CLASSIFICATION_IDS}
    expected |= {dataset_id: [EvalMetric.macro_auroc] for dataset_id in MULTI_LABEL_IDS}
    for dataset in MOLECULENET_CATALOG.datasets:
        assert dataset.metrics() == expected[dataset.dataset_id]
    multi_target_regression = MoleculeNetDataset.model_validate(
        _dataset(labels=[{"name": "first"}, {"name": "second"}])
    )
    assert multi_target_regression.metrics() == regression


@pytest.mark.parametrize(
    ("datasets", "source_files", "message"),
    [
        ([_dataset(), _dataset()], {"toy.csv": PIN}, "duplicate dataset_id"),
        ([_dataset(label_column="value")], {"toy.csv": PIN}, "label_column"),
        ([_dataset()], {"other.csv": PIN}, "have no pinned sha256"),
    ],
    ids=["duplicate id", "unknown field", "unpinned file"],
)
def test_the_catalog_rejects_an_inconsistent_entry(
    datasets: list[dict[str, Any]], source_files: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _catalog(datasets, source_files)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([{"name": "same"}, {"name": "same", "source_column": "other"}], "label names"),
        ([{"name": "first", "source_column": "value"}, {"name": "value"}], "columns"),
    ],
    ids=["names", "source columns"],
)
def test_a_dataset_rejects_duplicate_labels(
    labels: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MoleculeNetDataset.model_validate(_dataset(labels=labels))


# ------------------------------------------------------------ source files


def test_verify_source_file_checks_the_pin(tmp_path: Path) -> None:
    pin = _write_csv(tmp_path, "toy.csv", pd.DataFrame({"smiles": ["CCO"]}))
    assert verify_source_file(tmp_path, "toy.csv", pin).sha256 == pin
    with pytest.raises(MoleculeNetPreparationError, match=r"but datasets\.yaml pins"):
        verify_source_file(tmp_path, "toy.csv", PIN)
    with pytest.raises(MoleculeNetPreparationError, match="run with --download"):
        verify_source_file(tmp_path, "missing.csv", PIN)


def test_download_fetches_only_missing_files(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    pin = _write_csv(origin, "toy.csv", pd.DataFrame({"smiles": ["CCO"]}))
    raw_root = tmp_path / "raw"
    download_missing_file(origin.as_uri() + "/", raw_root, "toy.csv")
    assert sha256_of_file(raw_root / "toy.csv") == pin

    (raw_root / "toy.csv").write_text("kept as is\n")
    download_missing_file(origin.as_uri() + "/", raw_root, "toy.csv")
    assert (raw_root / "toy.csv").read_text() == "kept as is\n"

    with pytest.raises(MoleculeNetPreparationError, match="downloading"):
        download_missing_file(origin.as_uri() + "/", raw_root, "absent.csv")
    assert sorted(path.name for path in raw_root.iterdir()) == ["toy.csv"]


def test_ensure_source_files_downloads_then_checks_the_pin(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _write_csv(origin, "toy.csv", pd.DataFrame({"smiles": ["CCO"]}))
    catalog = _catalog([_dataset()], {"toy.csv": PIN}, origin.as_uri() + "/")
    config = MoleculeNetPreparerConfig(
        raw_root=tmp_path / "raw", catalog=catalog, download=True
    )
    with pytest.raises(MoleculeNetPreparationError, match=r"but datasets\.yaml pins"):
        ensure_source_files(config)
    assert (tmp_path / "raw" / "toy.csv").is_file()


# ------------------------------------------------------------ source tables


def test_read_source_table_reads_columns_by_name(tmp_path: Path) -> None:
    """Like tox21, the SMILES column comes after the labels and an id column."""
    frame = pd.DataFrame(
        {
            "second": [1, None, 0],
            "first": [0, 1, None],
            "mol_id": ["a", "b", "c"],
            "smiles": ["CCO", "CCCO", "CCCCO"],
        }
    )
    _write_csv(tmp_path, "toy.csv.gz", frame)
    dataset = MoleculeNetDataset.model_validate(
        _dataset(
            source_file="toy.csv.gz",
            task_type="classification",
            labels=[{"name": "renamed", "source_column": "first"}, {"name": "second"}],
        )
    )
    source = read_source_table(tmp_path, dataset)
    assert source.raw_smiles == ["CCO", "CCCO", "CCCCO"]
    assert list(source.labels) == ["renamed", "second"]
    assert source.labels["renamed"][:2] == [0.0, 1.0]
    assert pd.isna(source.labels["renamed"][2])
    assert source.labels["second"][0] == 1.0


@pytest.mark.parametrize(
    ("frame", "source_rows", "message"),
    [
        (pd.DataFrame({"smiles": ["CCO"], "value": [1.0]}), 2, "1 rows, toy expects 2"),
        (pd.DataFrame({"SMILES": ["CCO"], "value": [1.0]}), 1, r"\['smiles'\]"),
        (pd.DataFrame({"smiles": ["CCO"], "value": ["high"]}), 1, "non-numeric"),
    ],
    ids=["row count", "missing column", "non-numeric label"],
)
def test_read_source_table_rejects_a_bad_file(
    tmp_path: Path, frame: pd.DataFrame, source_rows: int, message: str
) -> None:
    _write_csv(tmp_path, "toy.csv", frame)
    dataset = MoleculeNetDataset.model_validate(_dataset(source_rows=source_rows))
    with pytest.raises(MoleculeNetPreparationError, match=message):
        read_source_table(tmp_path, dataset)


def test_read_source_table_skips_blank_lines(tmp_path: Path) -> None:
    """The S3 copy of HIV.csv has a blank line after every row."""
    (tmp_path / "toy.csv").write_bytes(
        b"smiles,value\r\n\nCCO,1\r\n\nCCCO,2\r\n\nCCCCO,3\r\n\n"
    )
    source = read_source_table(tmp_path, MoleculeNetDataset.model_validate(_dataset()))
    assert source.raw_smiles == ["CCO", "CCCO", "CCCCO"]


# ------------------------------------------- end to end on synthetic csvs


def _synthetic_sources(raw_root: Path) -> MoleculeNetCatalog:
    """A regression file and a multi-label classification file, with their pins.

    ``regression.csv`` (24 rows): the twenty ring molecules, one replicate of
    the first (merged by mean), one invalid SMILES, one isotope-labelled SMILES
    (filtered) and one blank SMILES (invalid).

    ``multilabel.csv.gz`` (25 rows, SMILES column last): the twenty ring
    molecules with one blank cell in each label; a replicate of ring 0 that
    ties on ``first`` only (kept, ``first`` missing); two replicates of ring 1,
    one blank, that tie on both labels (dropped as a tie); and two molecules
    without any label (dropped as missing).
    """
    regression_extras = ["c1ccccc1C", "not a molecule", "[11CH3]c1ccccc1", None]
    regression = pd.DataFrame(
        {
            "smiles": [*RING_MOLECULES, *regression_extras],
            "measured value": [float(index) for index in range(24)],
        }
    )
    missing = float("nan")
    first = [float(index % 2) for index in range(20)]
    second = [float((index + 1) % 2) for index in range(20)]
    first[5] = missing
    second[6] = missing
    multilabel_extras = [*RING_MOLECULES[:2], RING_MOLECULES[1], "CCO", "CCCO"]
    multilabel = pd.DataFrame(
        {
            "first": [*first, 1.0, 0.0, missing, missing, missing],
            "second": [*second, 1.0, 1.0, missing, missing, missing],
            "smiles": [*RING_MOLECULES, *multilabel_extras],
        }
    )
    return _catalog(
        [
            _dataset(
                dataset_id="toy_regression",
                source_file="regression.csv",
                source_rows=24,
                labels=[{"name": "toy_value", "source_column": "measured value"}],
            ),
            _dataset(
                dataset_id="toy_multilabel",
                source_file="multilabel.csv.gz",
                source_rows=25,
                task_type="classification",
                labels=[{"name": "first"}, {"name": "second"}],
            ),
        ],
        {
            "regression.csv": _write_csv(raw_root, "regression.csv", regression),
            "multilabel.csv.gz": _write_csv(raw_root, "multilabel.csv.gz", multilabel),
        },
    )


def _assert_the_scaffold_test_fold_is_fixed(
    table: pd.DataFrame, config: MoleculeNetPreparerConfig
) -> None:
    """Every split column has the DeepChem scaffold test fold; valid varies by seed."""
    expected_test = scaffold_test_mask(
        list(table["isomeric_smiles"]),
        config.scaffold_train_fraction,
        config.scaffold_valid_fraction,
    ).tolist()
    assert any(expected_test)
    for column in config.split_columns():
        assert (table[column] == "test").tolist() == expected_test
    seeded_columns = config.split_columns()[1:]
    assert len({tuple(table[column] == "valid") for column in seeded_columns}) > 1


def test_run_writes_valid_bundles(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    catalog = _synthetic_sources(raw_root)
    config = MoleculeNetPreparerConfig(
        raw_root=raw_root,
        bundle_root=tmp_path / "bundles",
        catalog=catalog,
        seeds=[1, 2, 3],
    )
    regression_report, multilabel_report = run(config)

    # ---- regression
    assert regression_report.counts.source_molecules == 24
    assert regression_report.counts.dropped == {
        INVALID_SMILES: 2,
        SMILES_FILTER: 1,
        DUPLICATE_SMILES: 1,
        DUPLICATE_LABEL_TIE: 0,
        MISSING_LABEL: 0,
    }
    assert regression_report.counts.final_rows == 20
    assert regression_report.rows_with_replicates == 1

    bundle = read_bundle(regression_report.directory)
    table = bundle.table
    assert list(table.columns) == [
        *"stereoisomer_id molecule_id enantiomer_of".split(),
        *"isomeric_smiles nonisomeric_smiles toy_value".split(),
        *config.split_columns(),
        MEASUREMENT_COUNT_COLUMN,
    ]
    assert table["toy_value"].iloc[0] == pytest.approx((0.0 + 20.0) / 2)
    assert list(table[MEASUREMENT_COUNT_COLUMN]) == [2.0] + [1.0] * 19
    _assert_the_scaffold_test_fold_is_fixed(table, config)
    assert (table["split"] == "test").sum() == 2
    assert bundle.spec.source_kind == "moleculenet_deepchem_s3"
    assert bundle.spec.evaluation is not None
    assert bundle.spec.evaluation.metrics[0] is EvalMetric.rmse
    provenance = bundle.provenance
    assert provenance.notices == [AGGREGATION_NOTICE, config.split_notice()]
    assert (
        provenance.source.files["regression.csv"].sha256
        == catalog.source_files["regression.csv"]
    )
    assert set(provenance.source.package_versions) == {"rdkit"}
    assert provenance.smiles_filter is not None
    assert provenance.smiles_filter.dedupe is False

    # ---- multi-label
    counts = multilabel_report.counts
    assert counts.source_molecules == 25
    assert counts.dropped == {
        INVALID_SMILES: 0,
        SMILES_FILTER: 0,
        DUPLICATE_SMILES: 3,
        DUPLICATE_LABEL_TIE: 1,
        MISSING_LABEL: 2,
    }
    assert counts.final_rows == 19
    assert counts.per_label_non_null == {"first": 17, "second": 18}
    multilabel = read_bundle(multilabel_report.directory)
    assert multilabel.spec.evaluation is not None
    assert multilabel.spec.evaluation.metrics == [EvalMetric.macro_auroc]
    assert [label.name for label in multilabel.spec.labels] == ["first", "second"]
    table = multilabel.table
    assert RING_MOLECULES[1] not in set(table["isomeric_smiles"])
    partial_tie_row = table[table["isomeric_smiles"] == RING_MOLECULES[0]].iloc[0]
    assert pd.isna(partial_tie_row["first"])
    assert partial_tie_row["second"] == 1.0
    assert partial_tie_row[MEASUREMENT_COUNT_COLUMN] == 2.0
    assert set(table["first"].dropna()) <= {0.0, 1.0}


def test_prepare_dataset_refuses_a_classification_label_of_two(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"smiles": RING_MOLECULES[:3], "value": [2.0, 0.0, 1.0]})
    pin = _write_csv(tmp_path, "toy.csv", frame)
    catalog = _catalog([_dataset(task_type="classification")], {"toy.csv": pin})
    config = MoleculeNetPreparerConfig(
        raw_root=tmp_path, bundle_root=tmp_path / "bundles", catalog=catalog
    )
    with pytest.raises(LabelAggregationError, match="neither 0 nor 1"):
        prepare_dataset(
            catalog.datasets[0],
            config,
            verify_source_file(tmp_path, "toy.csv", pin),
            PREPARER,
        )
    assert not (tmp_path / "bundles" / "toy").exists()


def test_run_checks_every_pin_before_writing(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    catalog = _synthetic_sources(raw_root)
    (raw_root / "multilabel.csv.gz").write_bytes(b"tampered")
    config = MoleculeNetPreparerConfig(
        raw_root=raw_root, bundle_root=tmp_path / "bundles", catalog=catalog
    )
    with pytest.raises(MoleculeNetPreparationError, match=r"but datasets\.yaml pins"):
        run(config)
    assert not (tmp_path / "bundles").exists()


def test_main_fails_without_raw_files_or_on_an_unknown_dataset(
    tmp_path: Path,
) -> None:
    bundle_root = str(tmp_path / "bundles")
    assert main(["--raw-root", str(tmp_path), "--bundle-root", bundle_root]) == 1
    assert not (tmp_path / "bundles").exists()
    assert main(["--bundle-root", bundle_root, "--only", "muv"]) == 1


# --------------------------------------------------- end to end, real data


@requires_raw_freesolv
def test_prepare_the_smallest_real_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pin = MOLECULENET_CATALOG.source_files["SAMPL.csv"]
    assert sha256_of_file(RAW_ROOT / "SAMPL.csv") == pin
    exit_code = main(["--bundle-root", str(tmp_path / "bundles"), "--only", "freesolv"])
    assert exit_code == 0
    assert "freesolv" in capsys.readouterr().out

    bundle = read_bundle(tmp_path / "bundles" / "freesolv")
    counts = bundle.provenance.counts
    assert counts.source_molecules == 642
    assert counts.final_rows + sum(counts.dropped.values()) == 642
    config = MoleculeNetPreparerConfig()
    assert bundle.spec.split_columns() == config.split_columns()
    assert len(config.split_columns()) == 6
    _assert_the_scaffold_test_fold_is_fixed(bundle.table, config)
    assert bundle.table["hydration_free_energy"].notna().all()
    assert bundle.provenance.source.files["SAMPL.csv"].sha256 == pin
    assert bundle.provenance.preparer.script == (
        "preparers/moleculenet/remedi_prepare_moleculenet/prepare.py"
    )
