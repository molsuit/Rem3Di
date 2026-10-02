"""The bundle and dataset format (``BENCHMARK_DATA_FORMAT.md`` §10).

Covers the :class:`DatasetSpec` validators, the disk round trip of the three
table files, every invariant ``validate_table`` (1-7) and ``structure_problems``
(8-10) check, the atomic write, the content hash, discovery, the data identity
a descriptor cache keys on, and ``PreparerRecord.for_script``.

The bundle package is torch-free, so this module also runs without the
repository conftest::

    uv run --no-sync pytest --noconftest tests/test_bundle_format.py -q
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd
import pydantic
import pytest
import yaml
from ase import Atoms
from rdkit import Chem

from remedi.data_handling.bundle import (
    DATASET_CONFIG_FILENAME,
    FORMAT_VERSION,
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    TABLE_FILENAME,
    Bundle,
    BundleValidationError,
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    LabelColumn,
    PreparerRecord,
    assign_identity,
    content_hash_of_table,
    count_stereoisomer_straddling_constitutions,
    discover_bundles,
    discover_datasets,
    is_dataset_directory,
    metrics_with_headline,
    mirror_isomeric_smiles,
    normalize_table,
    read_bundle,
    read_provenance,
    read_spec,
    read_table,
    stereochemistry_from_frame,
    structure_problems,
    structures_identity,
    tetrahedral_stereo_smiles,
    validate_table,
    write_bundle,
    write_table_files,
)
from remedi.data_handling.bundle import bundle as bundle_module
from remedi.data_handling.chemistry.geometry import GeometryLimits
from remedi.data_handling.dataset.tasks import TaskScope, TaskType

from .helpers.bundle_fixtures import (
    GEOMETRY_LIMITS,
    embed_frame,
    make_bundle,
    make_bundle_spec,
    make_dataset_spec,
    make_dataset_table,
    make_provenance,
    write_smiles_bundle,
)

THREE_FILES = sorted([SPEC_FILENAME, TABLE_FILENAME, PROVENANCE_FILENAME])

#: An enantiomer pair and an achiral molecule, for the dataset-kind cases.
CHIRAL_DATASET_SMILES = ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CCO"]


def has_problem(problems: list[str], prefix: str, fragment: str = "") -> bool:
    return any(
        problem.startswith(prefix) and fragment in problem for problem in problems
    )


def make_small_dataset(
    smiles_values: list[str] | None = None, *, smiles: bool = True
) -> tuple[DatasetSpec, pd.DataFrame]:
    """A dataset spec and its table: one structure per SMILES, one label."""
    smiles_values = smiles_values or list(CHIRAL_DATASET_SMILES)
    spec = make_dataset_spec(
        dataset_id="small",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
    )
    splits = {"split": ["train"] * (len(smiles_values) - 1) + ["test"]}
    table = make_dataset_table(
        spec, smiles_values, np.arange(len(smiles_values), dtype=float), splits
    )
    if not smiles:
        spec = spec.model_copy(update={"smiles": False})
        table = table.drop(columns=["isomeric_smiles", "nonisomeric_smiles"])
    return spec, normalize_table(table, spec)


# ------------------------------------------------------------------------ spec


def test_a_spec_defaults_to_format_version_1_and_no_smiles() -> None:
    spec = DatasetSpec(dataset_id="minimal", source_kind="synthetic")
    assert spec.format_version == FORMAT_VERSION == 1
    assert spec.smiles is False
    assert spec.labels == []
    assert spec.evaluation is None
    assert spec.has_structures is False
    assert spec.expected_columns() == [
        "stereoisomer_id",
        "molecule_id",
        "enantiomer_of",
    ]


def test_a_spec_refuses_any_other_format_version() -> None:
    with pytest.raises(pydantic.ValidationError):
        DatasetSpec.model_validate(
            {"dataset_id": "x", "source_kind": "synthetic", "format_version": 2}
        )


@pytest.mark.parametrize(
    "smiles, geometry_origin, expected",
    [
        (False, None, ["stereoisomer_id", "molecule_id", "enantiomer_of"]),
        (
            True,
            None,
            [
                "stereoisomer_id",
                "molecule_id",
                "enantiomer_of",
                "isomeric_smiles",
                "nonisomeric_smiles",
            ],
        ),
        (
            False,
            "source",
            [
                "structure_id",
                "stereoisomer_id",
                "molecule_id",
                "enantiomer_of",
                "total_charge",
                "multiplicity",
            ],
        ),
        (
            True,
            "etkdg_mmff",
            [
                "structure_id",
                "stereoisomer_id",
                "molecule_id",
                "enantiomer_of",
                "isomeric_smiles",
                "nonisomeric_smiles",
                "total_charge",
                "multiplicity",
            ],
        ),
    ],
)
def test_fixed_columns_depend_on_smiles_and_on_the_kind(
    smiles: bool, geometry_origin: str | None, expected: list[str]
) -> None:
    spec = DatasetSpec.model_validate(
        {
            "dataset_id": "x",
            "smiles": smiles,
            "geometry_origin": geometry_origin,
            "source_kind": "synthetic",
        }
    )
    assert spec.fixed_columns() == expected
    assert spec.has_structures is (geometry_origin is not None)


def test_expected_columns_is_the_declared_order() -> None:
    spec = DatasetSpec(
        dataset_id="ordered",
        smiles=True,
        labels=[LabelColumn(name="rs", task_type=TaskType.classification)],
        extra_columns=["n_chiral", "csd_code"],
        evaluation=EvaluationSpec(
            metrics=[EvalMetric.auroc],
            split_columns=["split", "split__scaffold_s0"],
            default_split="split__scaffold_s0",
            split_group="molecule_id",
        ),
        source_kind="synthetic",
    )
    assert spec.expected_columns() == [
        "stereoisomer_id",
        "molecule_id",
        "enantiomer_of",
        "isomeric_smiles",
        "nonisomeric_smiles",
        "rs",
        "split",
        "split__scaffold_s0",
        "n_chiral",
        "csd_code",
    ]
    assert spec.with_structures("etkdg_mmff").expected_columns()[:1] == ["structure_id"]


def test_a_corpus_has_labels_but_no_split_columns() -> None:
    spec = make_bundle_spec(evaluation=False)
    assert spec.split_columns() == []
    assert spec.label_names() == ["activity", "logp"]
    assert "split" not in spec.expected_columns()


def test_with_structures_turns_a_bundle_spec_into_a_dataset_spec() -> None:
    bundle_spec = make_bundle_spec()
    dataset_spec = bundle_spec.with_structures("etkdg_mmff")
    assert dataset_spec.geometry_origin == "etkdg_mmff"
    assert dataset_spec.has_structures
    assert bundle_spec.geometry_origin is None  # a copy, not a mutation


def test_with_structures_validates_the_dataset_spec() -> None:
    """A bundle extra column named like a dataset fixed column cannot become a dataset."""
    bundle_spec = make_bundle_spec().model_copy(
        update={"extra_columns": ["total_charge"]}
    )
    with pytest.raises(ValueError, match="total_charge"):
        bundle_spec.with_structures("etkdg_mmff")


@pytest.mark.parametrize(
    "update",
    [
        # duplicate label names
        {
            "labels": [
                {"name": "activity", "task_type": "classification"},
                {"name": "activity", "task_type": "regression"},
            ]
        },
        # a label colliding with a fixed column
        {"labels": [{"name": "molecule_id", "task_type": "regression"}]},
        # a label colliding with a SMILES column
        {"labels": [{"name": "isomeric_smiles", "task_type": "regression"}]},
        # an extra column colliding with a label
        {"extra_columns": ["logp"]},
        # an extra column colliding with a split column
        {"extra_columns": ["split"]},
        # the same extra column twice
        {"extra_columns": ["weight", "weight"]},
        # an evaluation block with nothing to score
        {"labels": []},
        # unknown field (extra="forbid")
        {"csv_name": "esol.csv"},
        # an unknown geometry origin
        {"geometry_origin": "xtb"},
        # an empty source_kind
        {"source_kind": ""},
    ],
)
def test_spec_validators_refuse(update: dict) -> None:
    document = make_bundle_spec().model_dump(mode="json")
    with pytest.raises(pydantic.ValidationError):
        DatasetSpec.model_validate({**document, **update})


def test_a_dataset_fixed_column_collides_with_a_declared_column() -> None:
    """``total_charge`` is only fixed on a dataset, so only there does it collide."""
    document = {
        "dataset_id": "x",
        "extra_columns": ["total_charge"],
        "source_kind": "synthetic",
    }
    assert DatasetSpec.model_validate(document).extra_columns == ["total_charge"]
    with pytest.raises(pydantic.ValidationError):
        DatasetSpec.model_validate({**document, "geometry_origin": "source"})


@pytest.mark.parametrize(
    "update",
    [
        {"split_columns": ["split__a"]},  # default_split "split" absent
        {"split_columns": ["split", "folds"]},  # bad split column name
        {"split_columns": ["split", "split__"]},  # empty variant
        {"metrics": []},
        {"split_columns": []},
        {"split_group": "structure_id"},
        {"metrics": ["accuracy"]},
        {"unknown": 1},
    ],
)
def test_evaluation_validators_refuse(update: dict) -> None:
    document = {
        "metrics": ["MAE"],
        "split_columns": ["split"],
        "default_split": "split",
        "split_group": "molecule_id",
    }
    with pytest.raises(pydantic.ValidationError):
        EvaluationSpec.model_validate({**document, **update})


def test_label_column_validators() -> None:
    with pytest.raises(ValueError):  # multiclass without n_classes
        LabelColumn(name="chirality_class", task_type=TaskType.multiclass)
    with pytest.raises(ValueError):  # n_classes below 2
        LabelColumn(name="chirality_class", task_type=TaskType.multiclass, n_classes=1)
    with pytest.raises(ValueError):  # n_classes on a non-multiclass label
        LabelColumn(name="logp", task_type=TaskType.regression, n_classes=3)
    with pytest.raises(ValueError):  # empty name
        LabelColumn(name="", task_type=TaskType.regression)


def test_class_names_are_only_valid_on_a_matching_multiclass_label() -> None:
    label = LabelColumn(
        name="chirality_class",
        task_type=TaskType.multiclass,
        n_classes=3,
        class_names=["achiral", "central", "axial"],
    )
    assert label.class_names == ["achiral", "central", "axial"]
    assert (
        LabelColumn(
            name="chirality_class", task_type=TaskType.multiclass, n_classes=3
        ).class_names
        is None
    )
    with pytest.raises(ValueError):  # too few names
        LabelColumn(
            name="chirality_class",
            task_type=TaskType.multiclass,
            n_classes=3,
            class_names=["achiral", "central"],
        )
    with pytest.raises(ValueError):  # class_names on a non-multiclass label
        LabelColumn(
            name="logp", task_type=TaskType.regression, class_names=["low", "high"]
        )


def test_task_set_maps_every_label_to_a_system_column() -> None:
    task_set = make_bundle_spec().task_set()
    assert [config.name for config in task_set.system_cols] == ["activity", "logp"]
    assert [config.task_type for config in task_set.system_cols] == [
        TaskType.classification,
        TaskType.regression,
    ]
    assert all(config.scope is TaskScope.system for config in task_set.system_cols)
    assert task_set.atom_cols == []


def test_eval_metric_values_are_the_yaml_strings() -> None:
    assert {member.name: member.value for member in EvalMetric} == {
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R2",
        "spearman": "Spearman",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "macro_auroc": "macro-AUROC",
        "balanced_accuracy": "balanced-accuracy",
        "macro_f1": "macro-F1",
        "macro_auroc_ovr": "macro-AUROC-OvR",
        "pair_ranking_accuracy": "pair-ranking-accuracy",
    }


def test_metrics_with_headline_puts_the_headline_first_without_repeats() -> None:
    assert metrics_with_headline(EvalMetric.rmse, TaskType.regression) == [
        EvalMetric.rmse,
        EvalMetric.mae,
        EvalMetric.spearman,
        EvalMetric.r2,
    ]
    assert metrics_with_headline(EvalMetric.balanced_accuracy, TaskType.multiclass) == [
        EvalMetric.balanced_accuracy
    ]


def test_a_spec_round_trips_through_yaml(tmp_path: Path) -> None:
    spec = make_bundle_spec(require_enantiomer_pairs=True)
    (tmp_path / SPEC_FILENAME).write_text(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True))
    )
    assert read_spec(tmp_path) == spec


# ------------------------------------------------------------------ identity


def test_identity_assignment_of_the_six_rows() -> None:
    table = make_bundle().table
    # the alanine pair: one constitution, two stereoisomers, each the other's mirror
    assert table.loc[0, "molecule_id"] == table.loc[1, "molecule_id"]
    assert table.loc[0, "enantiomer_of"] == table.loc[1, "stereoisomer_id"]
    assert table.loc[1, "enantiomer_of"] == table.loc[0, "stereoisomer_id"]
    # achiral, meso, partner-less and achiral rows have no mirror partner
    assert table.loc[2:, "enantiomer_of"].isna().all()
    assert table["stereoisomer_id"].tolist() == list(range(6))


def test_identity_table_frame_is_the_fixed_columns_of_a_smiles_bundle() -> None:
    frame = assign_identity(["Oc1ccccc1", "c1ccccc1O", "CCO"]).to_frame()
    assert list(frame.columns) == make_bundle_spec().fixed_columns()
    # two spellings of phenol collapse onto one stereoisomer
    assert frame["stereoisomer_id"].tolist() == [0, 0, 1]
    assert str(frame["enantiomer_of"].dtype) == "Int64"


def test_meso_compound_has_no_enantiomer() -> None:
    meso = Chem.MolToSmiles(Chem.MolFromSmiles("O[C@@H]([C@@H](O)C(O)=O)C(O)=O"))
    assert mirror_isomeric_smiles(meso) == meso
    assert assign_identity([meso]).enantiomer_of.isna().all()


def test_two_stereocentres_pair_only_exact_mirror_images() -> None:
    frame = assign_identity(
        [
            "C[C@H](O)[C@H](C)N",
            "C[C@@H](O)[C@@H](C)N",
            "C[C@H](O)[C@@H](C)N",
            "C[C@@H](O)[C@H](C)N",
        ]
    ).to_frame()
    assert frame["molecule_id"].nunique() == 1
    assert frame["stereoisomer_id"].nunique() == 4
    assert frame.loc[0, "enantiomer_of"] == frame.loc[1, "stereoisomer_id"]
    assert frame.loc[2, "enantiomer_of"] == frame.loc[3, "stereoisomer_id"]


# --------------------------------------------------------------- round trips


def test_a_bundle_round_trips_through_disk(tmp_path: Path) -> None:
    bundle = make_bundle()
    assert validate_table(bundle.spec, normalize_table(bundle.table, bundle.spec)) == []

    written = write_bundle(bundle, tmp_path / "synthetic6")

    directory = tmp_path / "synthetic6"
    assert sorted(path.name for path in directory.iterdir()) == THREE_FILES
    reloaded = read_bundle(directory)
    pd.testing.assert_frame_equal(
        reloaded.table, normalize_table(bundle.table, bundle.spec)
    )
    assert reloaded.spec == bundle.spec
    assert reloaded.provenance == written
    assert str(reloaded.table["enantiomer_of"].dtype) == "Int64"

    counts = reloaded.provenance.counts
    assert counts.final_rows == 6
    assert counts.per_split == {"train": 3, "valid": 1, "test": 1, "unassigned": 1}
    assert counts.per_label_non_null == {"activity": 4, "logp": 5}
    assert counts.stereoisomer_straddling_constitutions == 0
    # the preparer's own counts are kept
    assert counts.source_molecules == 6
    assert counts.dropped == {"element_gate": 2}
    outputs = reloaded.provenance.outputs
    assert outputs is not None
    assert outputs.table_parquet.rows == 6
    assert outputs.table_parquet.content_sha256 == content_hash_of_table(reloaded.table)
    assert outputs.structures_sha256 is None
    assert reloaded.provenance.prepared_at is not None
    assert reloaded.provenance.geometry_limits == GEOMETRY_LIMITS


def test_write_bundle_does_not_mutate_the_callers_provenance(tmp_path: Path) -> None:
    bundle = make_bundle()
    write_bundle(bundle, tmp_path / "synthetic6")
    assert bundle.provenance.outputs is None
    assert bundle.provenance.prepared_at is None


@pytest.mark.parametrize(
    "options",
    [{"smiles": False}, {"evaluation": False}, {"require_enantiomer_pairs": True}],
)
def test_every_bundle_variant_round_trips(tmp_path: Path, options: dict) -> None:
    directory = write_smiles_bundle(tmp_path, **options)
    reloaded = read_bundle(directory)
    assert list(reloaded.table.columns) == reloaded.spec.expected_columns()


def test_a_corpus_bundle_counts_no_splits(tmp_path: Path) -> None:
    reloaded = read_bundle(write_smiles_bundle(tmp_path, evaluation=False))
    assert reloaded.spec.evaluation is None
    assert reloaded.provenance.counts.per_split == {}
    assert reloaded.provenance.counts.per_label_non_null == {
        "activity": 4,
        "logp": 5,
    }


def test_write_bundle_normalizes_dtypes(tmp_path: Path) -> None:
    """int32 ids, a plain-object enantiomer_of and int labels are coerced."""
    bundle = make_bundle()
    bundle.table["stereoisomer_id"] = bundle.table["stereoisomer_id"].astype("int32")
    bundle.table["split"] = bundle.table["split"].astype(object)
    write_bundle(bundle, tmp_path / "synthetic6")
    reloaded = read_bundle(tmp_path / "synthetic6")
    assert reloaded.table["stereoisomer_id"].dtype == np.int64


def test_write_bundle_refuses_a_spec_with_structures(tmp_path: Path) -> None:
    bundle = make_bundle()
    bundle.spec = bundle.spec.with_structures("etkdg_mmff")
    with pytest.raises(BundleValidationError, match="geometry_origin"):
        write_bundle(bundle, tmp_path / "synthetic6")
    assert not (tmp_path / "synthetic6").exists()


def test_write_bundle_refuses_an_invalid_table_and_writes_nothing(
    tmp_path: Path,
) -> None:
    bundle = make_bundle()
    bundle.table["split"] = ["train", "valid", "test", "train", "test", "train"]
    with pytest.raises(BundleValidationError) as raised:
        write_bundle(bundle, tmp_path / "broken")
    assert has_problem(raised.value.problems, "6:")
    assert raised.value.directory == tmp_path / "broken"
    assert not (tmp_path / "broken").exists()


def test_read_bundle_rejects_a_tampered_content_hash(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    provenance_path = directory / PROVENANCE_FILENAME
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["table.parquet"]["content_sha256"] = "0" * 64
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert any("content_sha256" in problem for problem in raised.value.problems)


def test_read_bundle_rejects_a_tampered_table(tmp_path: Path) -> None:
    """A changed label value moves the content hash away from the record."""
    directory = write_smiles_bundle(tmp_path)
    table = pd.read_parquet(directory / TABLE_FILENAME)
    table.loc[0, "logp"] = 42.0
    table.to_parquet(directory / TABLE_FILENAME, index=False)
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert any("content_sha256" in problem for problem in raised.value.problems)


def test_read_bundle_rejects_a_wrong_row_count(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    provenance_path = directory / PROVENANCE_FILENAME
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["table.parquet"]["rows"] = 7
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError, match="7 rows"):
        read_bundle(directory)


def test_read_bundle_rejects_a_provenance_without_outputs(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    provenance_path = directory / PROVENANCE_FILENAME
    document = yaml.safe_load(provenance_path.read_text())
    del document["outputs"]
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError, match="no outputs"):
        read_bundle(directory)


def test_read_bundle_rejects_an_invariant_violated_on_disk(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    table = pd.read_parquet(directory / TABLE_FILENAME)
    table.loc[5, "split"] = "TEST"
    table.to_parquet(directory / TABLE_FILENAME, index=False)
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert has_problem(raised.value.problems, "5:")


@pytest.mark.parametrize("version", [0, 2, None, "1"])
def test_read_bundle_rejects_another_format_version(
    tmp_path: Path, version: object
) -> None:
    directory = write_smiles_bundle(tmp_path)
    spec_path = directory / SPEC_FILENAME
    document = yaml.safe_load(spec_path.read_text())
    document["format_version"] = version
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert any("format_version" in problem for problem in raised.value.problems)


def test_read_spec_rejects_a_spec_that_is_not_a_mapping(tmp_path: Path) -> None:
    (tmp_path / SPEC_FILENAME).write_text("- just\n- a list\n")
    with pytest.raises(BundleValidationError, match="not a mapping"):
        read_spec(tmp_path)


@pytest.mark.parametrize("missing", THREE_FILES)
def test_read_bundle_reports_a_missing_file(tmp_path: Path, missing: str) -> None:
    directory = write_smiles_bundle(tmp_path)
    (directory / missing).unlink()
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        read_bundle(directory)


def test_read_table_reads_columns_without_validating(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    assert len(read_table(directory)) == 6
    subset = read_table(directory, columns=["stereoisomer_id", "split"])
    assert list(subset.columns) == ["stereoisomer_id", "split"]
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nowhere")


def test_read_provenance_alone(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    assert read_provenance(directory).dataset_id == "synthetic6"
    with pytest.raises(FileNotFoundError):
        read_provenance(tmp_path / "nowhere")


# ------------------------------------------------------------ atomic writes


def test_a_successful_write_leaves_no_staging_files(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    assert not [path for path in directory.iterdir() if path.name.startswith(".")]


def test_a_failed_write_leaves_no_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table is staged, then the provenance dump fails: nothing lands."""

    def failing_dump(model, path: Path, **dump_options) -> None:
        if path.name.startswith(f".{PROVENANCE_FILENAME}"):
            raise OSError("disk full")
        path.write_text("staged")

    monkeypatch.setattr(bundle_module, "_dump_yaml", failing_dump)
    with pytest.raises(OSError, match="disk full"):
        write_bundle(make_bundle(), tmp_path / "synthetic6")
    assert list((tmp_path / "synthetic6").iterdir()) == []


def test_a_failed_rewrite_keeps_the_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_smiles_bundle(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    def failing_to_parquet(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)
    with pytest.raises(OSError):
        write_bundle(make_bundle(), directory)
    monkeypatch.undo()

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    read_bundle(directory)


def test_write_table_files_records_a_structures_hash(tmp_path: Path) -> None:
    spec, table = make_small_dataset()
    written = write_table_files(
        tmp_path, spec, table, make_provenance("small"), structures_sha256="a" * 64
    )
    assert written.outputs is not None
    assert written.outputs.structures_sha256 == "a" * 64
    assert sorted(path.name for path in tmp_path.iterdir()) == THREE_FILES
    assert read_provenance(tmp_path) == written


# ------------------------------------------------------- table invariants


def _bundle_table() -> tuple[DatasetSpec, pd.DataFrame]:
    bundle = make_bundle()
    return bundle.spec, normalize_table(bundle.table, bundle.spec)


def _repeat_a_stereoisomer(table: pd.DataFrame) -> None:
    table.loc[5, "stereoisomer_id"] = 0


def _map_a_molecule_to_two_nonisomeric_smiles(table: pd.DataFrame) -> None:
    table.loc[5, "molecule_id"] = 0


def _break_enantiomer_symmetry(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array([1, None, None, None, None, None], "Int64")


def _make_enantiomer_reflexive(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array([0, None, None, None, None, None], "Int64")


def _dangle_an_enantiomer_pointer(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array([99, None, None, None, None, None], "Int64")


def _pair_across_two_molecules(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array([2, None, 0, None, None, None], "Int64")


def _bad_split_value(table: pd.DataFrame) -> None:
    table.loc[5, "split"] = "TEST"


def _leak_across_split_group(table: pd.DataFrame) -> None:
    table.loc[1, "split"] = "test"


def _leak_in_the_second_split_column(table: pd.DataFrame) -> None:
    table.loc[1, "split__random_s1"] = "valid"


def _reorder_columns(table: pd.DataFrame) -> None:
    logp = table.pop("logp")
    table.insert(3, "logp", logp)


def _drop_a_label_column(table: pd.DataFrame) -> None:
    table.drop(columns=["logp"], inplace=True)


def _add_an_undeclared_column(table: pd.DataFrame) -> None:
    table["weight"] = 1.0


def _wrong_id_dtype(table: pd.DataFrame) -> None:
    table["molecule_id"] = table["molecule_id"].astype("int32")


def _wrong_enantiomer_dtype(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = table["enantiomer_of"].astype("float64")


def _wrong_smiles_dtype(table: pd.DataFrame) -> None:
    table["isomeric_smiles"] = np.arange(len(table), dtype="int64")


def _wrong_split_dtype(table: pd.DataFrame) -> None:
    table["split__random_s1"] = np.zeros(len(table), dtype="int64")


def _wrong_label_dtype(table: pd.DataFrame) -> None:
    table["activity"] = table["activity"].fillna(0).astype("int64")


def _classification_label_out_of_range(table: pd.DataFrame) -> None:
    table.loc[0, "activity"] = 0.5


TABLE_MUTATIONS: list[tuple[FunctionType, str, str]] = [
    (_repeat_a_stereoisomer, "1:", "stereoisomer_id repeats"),
    (_map_a_molecule_to_two_nonisomeric_smiles, "2:", "nonisomeric_smiles"),
    (_break_enantiomer_symmetry, "4:", "not symmetric"),
    (_make_enantiomer_reflexive, "4:", "reflexive"),
    (_dangle_an_enantiomer_pointer, "4:", "absent"),
    (_pair_across_two_molecules, "4:", "two molecule_ids"),
    (_bad_split_value, "5:", "TEST"),
    (_leak_across_split_group, "6:", "split "),
    (_leak_in_the_second_split_column, "6:", "split__random_s1"),
    (_reorder_columns, "7:", "columns"),
    (_drop_a_label_column, "7:", "columns"),
    (_add_an_undeclared_column, "7:", "columns"),
    (_wrong_id_dtype, "7:", "molecule_id"),
    (_wrong_enantiomer_dtype, "7:", "enantiomer_of"),
    (_wrong_smiles_dtype, "7:", "isomeric_smiles"),
    (_wrong_split_dtype, "7:", "split__random_s1"),
    (_wrong_label_dtype, "7:", "activity"),
    (_classification_label_out_of_range, "7:", "0 or 1"),
]


@pytest.mark.parametrize(
    "mutation, prefix, fragment",
    TABLE_MUTATIONS,
    ids=[mutation.__name__.strip("_") for mutation, _, _ in TABLE_MUTATIONS],
)
def test_bundle_table_invariants_are_detected(
    mutation: Callable[[pd.DataFrame], None], prefix: str, fragment: str
) -> None:
    spec, table = _bundle_table()
    assert validate_table(spec, table) == []
    mutation(table)
    problems = validate_table(spec, table)
    assert has_problem(problems, prefix, fragment), problems


def test_a_stereoisomer_with_two_isomeric_smiles_fails_invariant_2() -> None:
    """Only a dataset can repeat a stereoisomer, so only there can its SMILES differ."""
    spec, table = make_small_dataset()
    repeated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    repeated["structure_id"] = np.arange(len(repeated), dtype="int64")
    repeated.loc[3, "isomeric_smiles"] = "C[C@@H](N)C(=O)O"
    repeated.loc[3, "split"] = "train"
    problems = validate_table(spec, repeated)
    assert has_problem(problems, "2:", "isomeric_smiles"), problems


def test_a_three_level_nesting_violation_is_detected() -> None:
    """Invariant 3: a stereoisomer_id spanning two molecule_ids (dataset rows)."""
    spec, table = make_small_dataset(["C[C@H](N)C(=O)O", "CCO", "CCN"], smiles=False)
    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    duplicated["structure_id"] = np.arange(len(duplicated), dtype="int64")
    duplicated.loc[3, "molecule_id"] = 1
    duplicated.loc[3, "split"] = "test"
    problems = validate_table(spec, duplicated)
    assert has_problem(problems, "3:"), problems


def test_a_null_split_value_fails_invariant_5() -> None:
    spec, table = _bundle_table()
    table["split"] = table["split"].astype(object)
    table.loc[5, "split"] = None
    assert has_problem(validate_table(spec, table), "5:", "<NA>")


def test_require_enantiomer_pairs_rejects_a_lone_row() -> None:
    bundle = make_bundle()
    spec = make_bundle_spec(require_enantiomer_pairs=True)
    problems = validate_table(spec, normalize_table(bundle.table, spec))
    assert has_problem(problems, "4:", "require_enantiomer_pairs"), problems
    paired = make_bundle(require_enantiomer_pairs=True)
    assert validate_table(paired.spec, normalize_table(paired.table, spec)) == []


def test_multiclass_values_must_be_integers_in_range() -> None:
    spec = DatasetSpec(
        dataset_id="chirality",
        labels=[
            LabelColumn(
                name="chirality_class", task_type=TaskType.multiclass, n_classes=3
            )
        ],
        source_kind="synthetic",
    )
    table = assign_identity(["CCO", "CCN", "CCC"]).to_frame()
    table = table.drop(columns=["isomeric_smiles", "nonisomeric_smiles"])
    table["chirality_class"] = [0.0, 2.0, np.nan]
    assert validate_table(spec, normalize_table(table, spec)) == []
    for bad_value in (3.0, -1.0, 1.5):
        table.loc[0, "chirality_class"] = bad_value
        assert has_problem(validate_table(spec, normalize_table(table, spec)), "7:")


def test_smiles_invariants_are_skipped_without_smiles() -> None:
    """Without ``smiles: true`` there are no SMILES columns to map ids onto."""
    bundle = make_bundle(smiles=False)
    table = normalize_table(bundle.table, bundle.spec)
    assert "isomeric_smiles" not in table.columns
    assert validate_table(bundle.spec, table) == []
    with_smiles = make_bundle().table
    problems = validate_table(bundle.spec, normalize_table(with_smiles, bundle.spec))
    assert has_problem(problems, "7:", "columns")


def test_split_invariants_are_skipped_without_an_evaluation_block() -> None:
    """A corpus has no split columns: duplicated folds cannot leak."""
    bundle = make_bundle(evaluation=False)
    table = normalize_table(bundle.table, bundle.spec)
    assert validate_table(bundle.spec, table) == []
    assert not [name for name in table.columns if name.startswith("split")]


def test_a_dataset_needs_a_dense_structure_id() -> None:
    spec, table = make_small_dataset()
    assert validate_table(spec, table) == []
    table["structure_id"] = table["structure_id"].to_numpy()[::-1]
    assert has_problem(validate_table(spec, table), "1:", "structure_id")


def test_a_dataset_may_repeat_a_stereoisomer() -> None:
    """Several conformers of one stereoisomer are several dataset rows."""
    spec, table = make_small_dataset()
    repeated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    repeated["structure_id"] = np.arange(len(repeated), dtype="int64")
    assert validate_table(spec, repeated) == []


def test_a_dataset_needs_float_charge_columns() -> None:
    spec, table = make_small_dataset()
    table["total_charge"] = table["total_charge"].astype("int64")
    assert has_problem(validate_table(spec, table), "7:", "total_charge")
    spec, table = make_small_dataset()
    table = table.drop(columns=["multiplicity"])
    assert has_problem(validate_table(spec, table), "7:", "columns")


def test_a_bundle_table_with_a_structure_id_is_refused() -> None:
    spec, table = _bundle_table()
    table.insert(0, "structure_id", np.arange(len(table), dtype="int64"))
    assert has_problem(validate_table(spec, table), "7:", "columns")


def test_straddling_constitutions_are_counted() -> None:
    bundle = make_bundle()
    table = bundle.table.copy()
    assert count_stereoisomer_straddling_constitutions(table, "split") == 0
    _leak_across_split_group(table)
    # The alanine pair is one constitution with two stereoisomers in two folds.
    assert count_stereoisomer_straddling_constitutions(table, "split") == 1
    with pytest.raises(KeyError):
        count_stereoisomer_straddling_constitutions(table, "split__absent")


def test_the_straddling_count_is_recorded_for_a_stereoisomer_split_group(
    tmp_path: Path,
) -> None:
    bundle = make_bundle()
    bundle.spec = bundle.spec.model_copy(
        update={
            "evaluation": bundle.spec.evaluation.model_copy(  # type: ignore[union-attr]
                update={"split_group": "stereoisomer_id"}
            )
        }
    )
    _leak_across_split_group(bundle.table)
    written = write_bundle(bundle, tmp_path / "synthetic6")
    assert written.counts.stereoisomer_straddling_constitutions == 1


# --------------------------------------------------- structure invariants


def test_matching_frames_pass_every_structure_invariant() -> None:
    spec, table = make_small_dataset()
    frames = [embed_frame(smiles) for smiles in table["isomeric_smiles"]]
    assert structure_problems(spec, table, frames, GEOMETRY_LIMITS) == []


def test_a_frame_count_mismatch_fails_invariant_8() -> None:
    spec, table = make_small_dataset()
    frames = [embed_frame(smiles) for smiles in table["isomeric_smiles"]]
    problems = structure_problems(spec, table, frames[:2], GEOMETRY_LIMITS)
    assert problems == ["8: 2 structures for 3 table rows"]


def test_reflected_geometry_fails_invariant_9() -> None:
    spec, table = make_small_dataset()
    frames = [embed_frame(smiles) for smiles in table["isomeric_smiles"]]
    frames[1].set_positions(frames[1].get_positions() * np.array([-1.0, 1.0, 1.0]))
    problems = structure_problems(spec, table, frames, GEOMETRY_LIMITS)
    assert has_problem(problems, "9:", "rows [1]"), problems


def test_a_frame_in_the_wrong_atom_order_fails_invariant_9() -> None:
    spec, table = make_small_dataset()
    frames = [embed_frame(smiles) for smiles in table["isomeric_smiles"]]
    frames[0] = frames[0][::-1]
    assert has_problem(
        structure_problems(spec, table, frames, GEOMETRY_LIMITS), "9:", "rows [0]"
    )


def test_invariant_9_does_not_run_without_smiles() -> None:
    spec, table = make_small_dataset(smiles=False)
    frames = [embed_frame(smiles) for smiles in CHIRAL_DATASET_SMILES]
    frames[1].set_positions(frames[1].get_positions() * np.array([-1.0, 1.0, 1.0]))
    assert structure_problems(spec, table, frames, GEOMETRY_LIMITS) == []


def _methane_like(symbols: str, distance: float = 1.1) -> Atoms:
    """A first atom at the origin and the others on the axes, ``distance`` away."""
    atoms = Atoms(symbols)
    positions = [[0.0, 0.0, 0.0]] + [
        [distance if axis == index % 3 else 0.0 for axis in range(3)]
        for index in range(len(atoms) - 1)
    ]
    atoms.set_positions(positions)
    return atoms


@pytest.mark.parametrize(
    "atoms, limits, reason",
    [
        (_methane_like("CH3"), GeometryLimits(max_atoms=3), "max_atoms"),
        (_methane_like("HH"), GeometryLimits(), "no heavy atom"),
        (
            _methane_like("CO"),
            GeometryLimits(reject_zero_hydrogen=True),
            "no hydrogen",
        ),
        (
            _methane_like("CCH"),
            GeometryLimits(min_hydrogen_heavy_ratio=1.0),
            "hydrogen/heavy ratio",
        ),
        (
            _methane_like("SiH3"),
            GeometryLimits(elements=["H", "C", "N", "O"]),
            "element outside",
        ),
        (
            _methane_like("CH3", distance=0.2),
            GeometryLimits(min_interatomic_distance=0.5),
            "min_interatomic_distance",
        ),
    ],
)
def test_each_geometry_limit_fails_invariant_10(
    atoms: Atoms, limits: GeometryLimits, reason: str
) -> None:
    spec, table = make_small_dataset(["CCO"], smiles=False)
    problems = structure_problems(spec, table, [atoms], limits)
    assert len(problems) == 1 and has_problem(problems, "10:", reason), problems


def test_many_offending_rows_are_summarised() -> None:
    spec, table = make_small_dataset(["CCO", "CCN", "CCC", "CO", "CN", "CS", "CF"])
    frames = [_methane_like("HH") for _ in range(len(table))]
    problems = structure_problems(spec, table, frames, GeometryLimits())
    assert has_problem(problems, "10:", "7 rows, first [0, 1, 2, 3, 4]"), problems


# --- invariant 9 only compares what the SMILES actually specifies -----------


@pytest.mark.parametrize(
    "isomeric_smiles",
    [
        # One assigned and one unassigned tetrahedral centre: the embedding
        # picks some configuration for the second, which must not count.
        "C[C@H](O)C(C)N",
        # Unspecified double bond next to an assigned centre: 3D perception
        # always yields E or Z, which must not count either.
        "CC=C[C@H](C)O",
        # Unspecified imine.
        "N=C(N)NC[C@@H]1COc2ccccc2O1",
    ],
)
def test_invariant_9_ignores_stereo_the_smiles_leaves_unspecified(
    isomeric_smiles: str,
) -> None:
    frame = embed_frame(isomeric_smiles, seed=7)
    assert stereochemistry_from_frame(isomeric_smiles, frame) == (
        tetrahedral_stereo_smiles(isomeric_smiles)
    )


def test_invariant_9_still_sees_an_inverted_assigned_centre() -> None:
    isomeric_smiles = "C[C@H](O)C(C)N"
    frame = embed_frame(isomeric_smiles, seed=7)
    frame.set_positions(frame.get_positions() * np.array([-1.0, 1.0, 1.0]))
    assert stereochemistry_from_frame(isomeric_smiles, frame) != (
        tetrahedral_stereo_smiles(isomeric_smiles)
    )


def test_stereochemistry_from_frame_returns_none_for_an_unusable_frame() -> None:
    assert stereochemistry_from_frame("not a smiles", Atoms("C")) is None
    assert stereochemistry_from_frame("C[C@H](N)C(=O)O", Atoms("C")) is None
    assert tetrahedral_stereo_smiles("not a smiles") is None


# ------------------------------------------------------------------ hashing


def test_parquet_file_hash_moves_but_content_hash_does_not(tmp_path: Path) -> None:
    table = make_bundle().table
    table.to_parquet(tmp_path / "a.parquet", index=False, compression="snappy")
    table.to_parquet(tmp_path / "b.parquet", index=False, compression="zstd")
    table.to_parquet(
        tmp_path / "c.parquet", index=False, compression="snappy", row_group_size=2
    )
    table.to_parquet(tmp_path / "d.parquet", index=False, compression="snappy")
    digests = {
        name: hashlib.sha256((tmp_path / f"{name}.parquet").read_bytes()).hexdigest()
        for name in "abcd"
    }
    assert digests["a"] == digests["d"]
    assert digests["a"] != digests["b"]
    assert digests["a"] != digests["c"]
    reread = [pd.read_parquet(tmp_path / f"{name}.parquet") for name in "abcd"]
    assert len({content_hash_of_table(frame) for frame in reread}) == 1


def test_content_hash_reacts_to_values_names_and_dtypes() -> None:
    table = make_bundle().table
    before = content_hash_of_table(table)
    assert content_hash_of_table(table.copy()) == before
    changed = table.copy()
    changed.loc[0, "logp"] = -2.9
    assert content_hash_of_table(changed) != before
    assert content_hash_of_table(table.rename(columns={"logp": "log_p"})) != before
    retyped = table.copy()
    retyped["molecule_id"] = retyped["molecule_id"].astype("int32")
    assert content_hash_of_table(retyped) != before


def test_content_hash_ignores_the_string_backend() -> None:
    table = make_bundle().table
    as_objects = table.copy()
    for column_name in ("isomeric_smiles", "nonisomeric_smiles", "split"):
        as_objects[column_name] = as_objects[column_name].astype(object)
    assert content_hash_of_table(as_objects) == content_hash_of_table(table)


def test_a_rewrite_of_the_same_bundle_records_the_same_content_hash(
    tmp_path: Path,
) -> None:
    first = write_bundle(make_bundle(), tmp_path / "a")
    second = write_bundle(make_bundle(), tmp_path / "b")
    assert first.outputs is not None and second.outputs is not None
    assert (
        first.outputs.table_parquet.content_sha256
        == second.outputs.table_parquet.content_sha256
    )


# ---------------------------------------------------------------- discovery


def _fake_dataset_directory(root: Path, dataset_id: str) -> Path:
    """A bundle directory plus a ``dataset_config.yaml``: a dataset to discovery."""
    directory = write_smiles_bundle(root, dataset_id=dataset_id)
    (directory / DATASET_CONFIG_FILENAME).write_text("contains_smiles: false\n")
    return directory


def test_discover_bundles_finds_nested_bundles_sorted(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path, dataset_id="zeta")
    write_smiles_bundle(tmp_path / "tdc", dataset_id="AMES")
    write_smiles_bundle(tmp_path, dataset_id="alpha")
    (tmp_path / "not_a_bundle").mkdir()
    _fake_dataset_directory(tmp_path, "a_dataset")

    assert discover_bundles(tmp_path) == [
        tmp_path / "alpha",
        tmp_path / "tdc" / "AMES",
        tmp_path / "zeta",
    ]
    assert discover_bundles(tmp_path / "missing") == []


def test_is_dataset_directory_needs_both_files(tmp_path: Path) -> None:
    bundle_directory = write_smiles_bundle(tmp_path)
    assert not is_dataset_directory(bundle_directory)
    (bundle_directory / DATASET_CONFIG_FILENAME).write_text("{}\n")
    assert is_dataset_directory(bundle_directory)
    zarr_only = tmp_path / "zarr_only"
    zarr_only.mkdir()
    (zarr_only / DATASET_CONFIG_FILENAME).write_text("{}\n")
    assert not is_dataset_directory(zarr_only)


def test_discover_datasets_finds_only_datasets_directly_under_the_root(
    tmp_path: Path,
) -> None:
    _fake_dataset_directory(tmp_path, "zeta")
    _fake_dataset_directory(tmp_path, "alpha")
    write_smiles_bundle(tmp_path, dataset_id="a_bundle")
    _fake_dataset_directory(tmp_path / "nested", "too_deep")
    (tmp_path / "status.yaml").write_text("n_tasks: 0\n")

    discovered = discover_datasets(tmp_path)

    assert [path.name for path, _ in discovered] == ["alpha", "zeta"]
    assert [spec.dataset_id for _, spec in discovered] == ["alpha", "zeta"]
    assert discover_datasets(tmp_path / "missing") == []


def test_discover_datasets_raises_on_an_unreadable_spec(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / SPEC_FILENAME).write_text("format_version: 99\n")
    (directory / DATASET_CONFIG_FILENAME).write_text("{}\n")
    with pytest.raises(BundleValidationError, match="format_version"):
        discover_datasets(tmp_path)


# ----------------------------------------------------------- data identity


def test_structures_identity_of_a_bundle_is_its_content_hash(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    outputs = read_provenance(directory).outputs
    assert outputs is not None
    assert structures_identity(directory) == outputs.table_parquet.content_sha256


def test_structures_identity_of_a_dataset_is_its_structures_hash(
    tmp_path: Path,
) -> None:
    spec, table = make_small_dataset()
    write_table_files(
        tmp_path, spec, table, make_provenance("small"), structures_sha256="b" * 64
    )
    assert structures_identity(tmp_path) == "b" * 64


def test_structures_identity_needs_a_provenance_with_outputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        structures_identity(tmp_path)
    (tmp_path / PROVENANCE_FILENAME).write_text(
        yaml.safe_dump(make_provenance("x").model_dump(mode="json", exclude_none=True))
    )
    with pytest.raises(BundleValidationError, match="no outputs"):
        structures_identity(tmp_path)


# --------------------------------------------------------------- preparer


def test_preparer_record_for_a_script_in_this_repository() -> None:
    record = PreparerRecord.for_script("molsuit/Rem3Di", Path(__file__))
    assert record.repo == "molsuit/Rem3Di"
    assert record.script.endswith("tests/test_bundle_format.py")
    assert record.git_sha is None or re.fullmatch(r"[0-9a-f]{40}", record.git_sha)


def test_preparer_record_for_a_script_in_a_nested_checkout(tmp_path: Path) -> None:
    """The nearest ``.git`` decides the relative path; an unreadable sha is None."""
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    script = repository / "preparers" / "tdc" / "prepare.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('prepare')\n")

    record = PreparerRecord.for_script("molsuit/remedi-data", script)

    assert record.script == "preparers/tdc/prepare.py"
    assert record.git_sha is None


def test_preparer_record_for_a_script_outside_any_checkout(tmp_path: Path) -> None:
    script = tmp_path / "loose_script.py"
    script.write_text("\n")
    record = PreparerRecord.for_script("somewhere", script)
    if not any((parent / ".git").exists() for parent in script.parents):
        assert record.script == "loose_script.py"


def test_a_bundle_dataclass_holds_the_three_parts() -> None:
    bundle = make_bundle()
    assert isinstance(bundle, Bundle)
    assert set(vars(bundle)) == {"spec", "table", "provenance"}
