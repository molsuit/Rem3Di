"""What every SMILES preparer shares (``bundle/preparation.py``)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from remedi.data_handling.bundle import (
    AGGREGATION_NOTICE,
    MEASUREMENT_COUNT_COLUMN,
    EvalMetric,
    Log10LabelTransform,
    PreparerRecord,
    SourceRecord,
    merge_source_rows,
    read_bundle,
)
from remedi.data_handling.bundle.preparation import (
    BundleReport,
    PreparationError,
    PreparerSettings,
    SourceLabel,
    format_report,
    preparer_main,
    seeded_split_notice,
    smiles_bundle_spec,
    smiles_bundle_table,
    smiles_evaluation_metrics,
    write_smiles_bundle,
)
from remedi.data_handling.dataset.tasks import TaskType

ETHANOL = "CCO"
PROPANOL = "CCCO"
BUTANOL = "CCCCO"
PREPARER = PreparerRecord(repo="molsuit/Rem3Di", script="prepare.py")


class ToySettings(PreparerSettings):
    """A preparer with a two-dataset catalog."""

    raw_root: Path = Path("raw")
    bundle_root: Path = Path("bundles")

    def dataset_ids(self) -> list[str]:
        return ["first", "second"]


def _merged(labels: dict[str, list[float]]):
    return merge_source_rows(
        [ETHANOL, PROPANOL, BUTANOL],
        labels,
        dict.fromkeys(labels, TaskType.regression),
        ToySettings().smiles_filter,
    )


def _spec(dataset_id: str = "first", label_names: tuple[str, ...] = ("y",)):
    return smiles_bundle_spec(
        dataset_id=dataset_id,
        description="a toy bundle",
        labels=[
            SourceLabel(name=name).label_column(TaskType.regression)
            for name in label_names
        ],
        metrics=[EvalMetric.mae],
        split_columns=["split"],
        source_kind="toy",
    )


# ----------------------------------------------------------------- settings


def test_settings_default_to_five_seeds_and_a_filter_without_dedupe() -> None:
    settings = ToySettings()
    assert settings.smiles_filter.dedupe is False
    assert settings.split_columns() == [
        "split",
        *(f"split__seed{s}" for s in range(1, 6)),
    ]
    assert settings.selected_dataset_ids() == ["first", "second"]
    selected = ToySettings(only=["second", "first"]).selected_dataset_ids()
    assert selected == ["first", "second"]  # catalog order


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"only": ["third"]}, "unknown dataset ids"),
        ({"seeds": [1, 1]}, r"duplicate seeds \['1'\]"),
    ],
)
def test_settings_reject_unknown_ids_and_duplicate_seeds(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ToySettings.model_validate(overrides)


def test_source_label_reads_its_own_name_unless_told_otherwise() -> None:
    assert SourceLabel(name="y").column == "y"
    label = SourceLabel(
        name="y", source_column="Y (raw)", transform=Log10LabelTransform()
    )
    assert label.column == "Y (raw)"
    assert label.label_column(TaskType.regression).transform.kind == "log10"


def test_seeded_split_notice_names_the_seeds_and_the_alias() -> None:
    notice = seeded_split_notice("the shipped fold", [3, 4], 0.25)
    assert notice.startswith("Test fold: the shipped fold")
    assert "(3, 4)" in notice
    assert "aliases split__seed3" in notice


# ---------------------------------------------------------- spec and table


def test_multi_label_classification_is_scored_by_macro_auroc_alone() -> None:
    assert smiles_evaluation_metrics(TaskType.classification, 3, EvalMetric.auroc) == [
        EvalMetric.macro_auroc
    ]
    single = smiles_evaluation_metrics(TaskType.classification, 1, EvalMetric.auroc)
    assert single[0] is EvalMetric.auroc
    regression = smiles_evaluation_metrics(TaskType.regression, 4, EvalMetric.mae)
    assert regression[0] is EvalMetric.mae and EvalMetric.rmse in regression


def test_spec_is_a_smiles_bundle_split_by_stereoisomer() -> None:
    spec = _spec()
    assert spec.smiles is True
    assert spec.extra_columns == [MEASUREMENT_COUNT_COLUMN]
    assert spec.evaluation is not None
    assert spec.evaluation.split_group == "stereoisomer_id"
    assert spec.evaluation.default_split == "split"


def test_table_has_identity_labels_splits_and_measurement_counts() -> None:
    merged = _merged({"y": [1.0, math.nan, 3.0], "z": [math.nan, 2.0, math.nan]})
    table = smiles_bundle_table(merged, {"split": ["train", "valid", "test"]})
    assert list(table.columns) == [
        "stereoisomer_id",
        "molecule_id",
        "enantiomer_of",
        "isomeric_smiles",
        "nonisomeric_smiles",
        "y",
        "z",
        "split",
        MEASUREMENT_COUNT_COLUMN,
    ]
    assert table["y"].isna().tolist() == [False, True, False]
    assert list(table[MEASUREMENT_COUNT_COLUMN]) == [1.0, 1.0, 1.0]


def test_table_rejects_an_unassigned_split_and_an_empty_label() -> None:
    merged = _merged({"y": [1.0, 2.0, 3.0]})
    with pytest.raises(PreparationError, match="unassigned rows"):
        smiles_bundle_table(merged, {"split": ["train", "unassigned", "test"]})
    merged = _merged({"y": [1.0, 2.0, 3.0], "z": [math.nan] * 3})
    with pytest.raises(PreparationError, match="no value at all"):
        smiles_bundle_table(merged, {"split": ["train", "valid", "test"]})


# ---------------------------------------------------------- write + report


def _write(tmp_path: Path) -> BundleReport:
    return write_smiles_bundle(
        spec=_spec(),
        merged=_merged({"y": [1.0, 2.0, 3.0]}),
        split_values={"split": ["train", "valid", "test"]},
        settings=ToySettings(bundle_root=tmp_path),
        preparer=PREPARER,
        source=SourceRecord(package_versions={"rdkit": "test"}),
        notices=["a source notice"],
        extra={"special": 7},
    )


def test_write_records_the_merge_and_the_filter_that_ran(tmp_path: Path) -> None:
    report = _write(tmp_path)
    bundle = read_bundle(tmp_path / "first")
    assert report.directory == tmp_path / "first"
    assert report.counts.final_rows == 3
    assert bundle.provenance.outputs is not None
    assert report.content_sha256 == (
        bundle.provenance.outputs.table_parquet.content_sha256
    )
    assert bundle.provenance.notices == [AGGREGATION_NOTICE, "a source notice"]
    assert bundle.provenance.smiles_filter is not None
    assert bundle.provenance.smiles_filter.dedupe is False
    assert bundle.provenance.counts.per_split == {"train": 1, "valid": 1, "test": 1}


def test_report_lists_drop_reasons_extras_and_the_hash(tmp_path: Path) -> None:
    report = _write(tmp_path)
    lines = format_report([report]).splitlines()
    header, row = lines[0].split(), lines[2].split()
    assert header[:3] == ["dataset_id", "metric", "labels"]
    assert "duplicate_smiles" in header and "special" in header
    assert row[0] == "first"
    assert row[header.index("special")] == "7"
    assert row[-1] == report.content_sha256[:14]
    assert format_report([]) == "(no datasets prepared)"


# ---------------------------------------------------------------------- cli


def test_main_prints_the_report_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[ToySettings] = []

    def run(settings: ToySettings) -> list[BundleReport]:
        seen.append(settings)
        return [_write(tmp_path)]

    exit_code = preparer_main(
        ["--bundle-root", str(tmp_path), "--only", "first", "--seeds", "7"],
        description="toy",
        settings_type=ToySettings,
        run=run,
        add_arguments=lambda parser: parser.add_argument(
            "--seeds", type=int, nargs="+"
        ),
        extra_settings=lambda arguments: {"seeds": arguments.seeds},
    )
    assert exit_code == 0
    assert seen[0].only == ["first"] and seen[0].seeds == [7]
    assert seen[0].raw_root == Path("raw")
    assert "first" in capsys.readouterr().out


def test_main_returns_one_on_a_preparation_error_or_bad_settings() -> None:
    def failing_run(settings: ToySettings) -> list[BundleReport]:
        raise PreparationError("the source is broken")

    for argv in ([], ["--only", "third"]):
        exit_code = preparer_main(
            argv, description="toy", settings_type=ToySettings, run=failing_run
        )
        assert exit_code == 1
