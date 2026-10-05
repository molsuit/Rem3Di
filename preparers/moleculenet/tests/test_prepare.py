"""Tests for the MoleculeNet-specific parts of the MoleculeNet preparer.

The shared mechanics (merge, table, spec, report, CLI) are tested with
``remedi.data_handling.bundle``. Here: the catalog, the pin check, the download
and the csv reader on tiny synthetic files, :func:`run` end to end on a
synthetic catalog whose pins are computed in the test, and one test on the
smallest real dataset (freesolv), skipped when the pinned raw file is absent.
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
    read_source_table,
    run,
    verify_source_file,
)

from remedi.data_handling.bundle import (
    AGGREGATION_NOTICE,
    INVALID_SMILES,
    EvalMetric,
    read_bundle,
    sha256_of_file,
)
from remedi.data_handling.chemistry.splits import scaffold_test_mask

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "moleculenet"
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
    assert set(catalog.source_files) == {
        dataset.source_file for dataset in catalog.datasets
    }
    assert len(catalog.source_files) == 9
    assert [label.column for label in by_id["esol"].labels] == [
        "measured log solubility in mols per litre"
    ]
    assert by_id["bace"].smiles_column == "mol"
    assert [label.name for label in by_id["clintox"].labels] == [
        "FDA_APPROVED",
        "CT_TOX",
    ]
    assert (len(by_id["sider"].labels), len(by_id["tox21"].labels)) == (27, 12)


def test_the_metrics_follow_the_moleculenet_convention() -> None:
    regression = [EvalMetric.rmse, EvalMetric.mae, EvalMetric.spearman, EvalMetric.r2]
    classification = [EvalMetric.auroc, EvalMetric.auprc]
    expected = {dataset_id: regression for dataset_id in REGRESSION_IDS}
    expected |= {dataset_id: classification for dataset_id in CLASSIFICATION_IDS}
    expected |= {dataset_id: [EvalMetric.macro_auroc] for dataset_id in MULTI_LABEL_IDS}
    for dataset in MOLECULENET_CATALOG.datasets:
        assert dataset.metrics() == expected[dataset.dataset_id]


@pytest.mark.parametrize(
    ("datasets", "source_files", "message"),
    [
        ([_dataset(), _dataset()], {"toy.csv": PIN}, "duplicate dataset_id"),
        ([_dataset()], {"other.csv": PIN}, "have no pinned sha256"),
        (
            [
                _dataset(
                    labels=[{"name": "same"}, {"name": "same", "source_column": "x"}]
                )
            ],
            {"toy.csv": PIN},
            "duplicate label names",
        ),
        (
            [_dataset(labels=[{"name": "first", "source_column": "y"}, {"name": "y"}])],
            {"toy.csv": PIN},
            "duplicate label source columns",
        ),
        ([_dataset(task_type="multiclass")], {"toy.csv": PIN}, "not supported"),
    ],
    ids=[
        "duplicate id",
        "unpinned file",
        "duplicate label",
        "duplicate source column",
        "unsupported task type",
    ],
)
def test_the_catalog_rejects_an_inconsistent_entry(
    datasets: list[dict[str, Any]], source_files: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _catalog(datasets, source_files)


def test_the_scaffold_fractions_must_leave_a_test_fold() -> None:
    with pytest.raises(ValueError, match="leave no test fold"):
        MoleculeNetPreparerConfig(
            scaffold_train_fraction=0.8, scaffold_valid_fraction=0.2
        )


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
            "smiles": ["CCO", None, "CCCCO"],
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
    # A blank SMILES cell is passed on as None for the filter to count.
    assert source.raw_smiles == ["CCO", None, "CCCCO"]
    assert list(source.labels) == ["renamed", "second"]
    assert source.labels["renamed"][:2] == [0.0, 1.0]
    assert pd.isna(source.labels["renamed"][2])


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


def _synthetic_catalog(raw_root: Path) -> MoleculeNetCatalog:
    """The twenty ring molecules plus a blank SMILES, as a pinned regression file."""
    frame = pd.DataFrame(
        {
            "smiles": [*RING_MOLECULES, None],
            "measured value": [float(index) for index in range(21)],
        }
    )
    return _catalog(
        [
            _dataset(
                dataset_id="toy_regression",
                source_file="regression.csv",
                source_rows=21,
                labels=[{"name": "toy_value", "source_column": "measured value"}],
            )
        ],
        {"regression.csv": _write_csv(raw_root, "regression.csv", frame)},
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


def test_run_freezes_the_scaffold_test_fold_and_records_the_pin(
    tmp_path: Path,
) -> None:
    catalog = _synthetic_catalog(tmp_path / "raw")
    config = MoleculeNetPreparerConfig(
        raw_root=tmp_path / "raw",
        bundle_root=tmp_path / "bundles",
        catalog=catalog,
        seeds=[1, 2, 3],
    )
    (report,) = run(config)
    assert report.counts.dropped[INVALID_SMILES] == 1
    bundle = read_bundle(report.directory)
    _assert_the_scaffold_test_fold_is_fixed(bundle.table, config)
    assert (bundle.table["split"] == "test").sum() == 2
    assert bundle.spec.source_kind == "moleculenet_deepchem_s3"
    provenance = bundle.provenance
    assert provenance.notices == [AGGREGATION_NOTICE, config.split_notice()]
    assert (
        provenance.source.files["regression.csv"].sha256
        == (catalog.source_files["regression.csv"])
    )
    assert set(provenance.source.package_versions) == {"rdkit"}


def test_run_checks_every_pin_before_writing(tmp_path: Path) -> None:
    catalog = _synthetic_catalog(tmp_path / "raw")
    (tmp_path / "raw" / "regression.csv").write_bytes(b"tampered")
    config = MoleculeNetPreparerConfig(
        raw_root=tmp_path / "raw", bundle_root=tmp_path / "bundles", catalog=catalog
    )
    with pytest.raises(MoleculeNetPreparationError, match=r"but datasets\.yaml pins"):
        run(config)
    assert not (tmp_path / "bundles").exists()
    assert main(["--raw-root", str(tmp_path), "--bundle-root", str(tmp_path)]) == 1


# --------------------------------------------------- end to end, real data


@requires_raw_freesolv
def test_prepare_the_smallest_real_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--bundle-root", str(tmp_path / "bundles"), "--only", "freesolv"])
    assert exit_code == 0
    assert "freesolv" in capsys.readouterr().out

    bundle = read_bundle(tmp_path / "bundles" / "freesolv")
    assert bundle.provenance.counts.source_molecules == 642
    config = MoleculeNetPreparerConfig()
    assert bundle.spec.split_columns() == config.split_columns()
    _assert_the_scaffold_test_fold_is_fixed(bundle.table, config)
    assert bundle.table["hydration_free_energy"].notna().all()
    assert (
        bundle.provenance.source.files["SAMPL.csv"].sha256
        == (MOLECULENET_CATALOG.source_files["SAMPL.csv"])
    )
