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
from typing import Literal

import numpy as np
import pandas as pd
import pydantic
import pytest
import yaml
from ase import Atoms
from rdkit import Chem

from remedi.data_handling.bundle import (
    DATASET_CONFIG_FILENAME,
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
IDENTITY_COLUMNS = ["stereoisomer_id", "molecule_id", "enantiomer_of"]
SMILES_COLUMNS = ["isomeric_smiles", "nonisomeric_smiles"]
CHARGE_COLUMNS = ["total_charge", "multiplicity"]

#: An enantiomer pair and an achiral molecule, for the dataset-kind cases.
CHIRAL_DATASET_SMILES = ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CCO"]

TableMutation = Callable[[pd.DataFrame], None]


def has_problem(problems: list[str], pattern: str) -> bool:
    """Whether some problem matches ``pattern`` from its start (the invariant number)."""
    return any(re.match(pattern, problem) for problem in problems)


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
        table = table.drop(columns=SMILES_COLUMNS)
    return spec, normalize_table(table, spec)


def bundle_table() -> tuple[DatasetSpec, pd.DataFrame]:
    bundle = make_bundle()
    return bundle.spec, normalize_table(bundle.table, bundle.spec)


def set_value(row: int, column: str, value: object) -> TableMutation:
    def mutate(table: pd.DataFrame) -> None:
        table.loc[row, column] = value

    return mutate


def retype(column: str, dtype: str) -> TableMutation:
    def mutate(table: pd.DataFrame) -> None:
        table[column] = table[column].astype(dtype)

    return mutate


def fill_integers(column: str) -> TableMutation:
    def mutate(table: pd.DataFrame) -> None:
        table[column] = np.zeros(len(table), dtype="int64")

    return mutate


def drop(column: str) -> TableMutation:
    return lambda table: table.drop(columns=[column], inplace=True)


def set_enantiomers(row_0: int, row_2: int | None = None) -> TableMutation:
    """Overwrite ``enantiomer_of`` of the six bundle rows: only rows 0 and 2 set."""

    def mutate(table: pd.DataFrame) -> None:
        values = [row_0, None, row_2, None, None, None]
        table["enantiomer_of"] = pd.array(values, "Int64")

    return mutate


def reflect(frame: Atoms) -> None:
    frame.set_positions(frame.get_positions() * np.array([-1.0, 1.0, 1.0]))


# ------------------------------------------------------------------------ spec


@pytest.mark.parametrize(
    "smiles, geometry_origin, expected",
    [
        (False, None, IDENTITY_COLUMNS),
        (True, None, IDENTITY_COLUMNS + SMILES_COLUMNS),
        (False, "source", ["structure_id", *IDENTITY_COLUMNS, *CHARGE_COLUMNS]),
        (
            True,
            "etkdg_mmff",
            ["structure_id", *IDENTITY_COLUMNS, *SMILES_COLUMNS, *CHARGE_COLUMNS],
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
        *IDENTITY_COLUMNS,
        *SMILES_COLUMNS,
        "rs",
        "split",
        "split__scaffold_s0",
        "n_chiral",
        "csd_code",
    ]
    assert spec.with_structures("etkdg_mmff").expected_columns()[:1] == ["structure_id"]
    corpus = make_bundle_spec(evaluation=False)
    assert corpus.split_columns() == []
    assert "split" not in corpus.expected_columns()


def test_with_structures_turns_a_bundle_spec_into_a_validated_dataset_spec() -> None:
    bundle_spec = make_bundle_spec()
    dataset_spec = bundle_spec.with_structures("etkdg_mmff")
    assert dataset_spec.geometry_origin == "etkdg_mmff"
    assert dataset_spec.has_structures
    assert bundle_spec.geometry_origin is None  # a copy, not a mutation
    # ``total_charge`` is only a fixed column on a dataset: a valid bundle
    # extra column, but one that cannot become a dataset.
    clashing = DatasetSpec.model_validate(
        {**bundle_spec.model_dump(), "extra_columns": ["total_charge"]}
    )
    with pytest.raises(ValueError, match="total_charge"):
        clashing.with_structures("etkdg_mmff")


@pytest.mark.parametrize(
    "update",
    [
        # duplicate label names
        {"labels": [{"name": "y", "task_type": "regression"}] * 2},
        # a label colliding with a fixed column, or with a SMILES column
        {"labels": [{"name": "molecule_id", "task_type": "regression"}]},
        {"labels": [{"name": "isomeric_smiles", "task_type": "regression"}]},
        # an extra column colliding with a label, a split column or itself
        {"extra_columns": ["logp"]},
        {"extra_columns": ["split"]},
        {"extra_columns": ["weight", "weight"]},
        # ``total_charge`` is a fixed column on a dataset only
        {"geometry_origin": "source", "extra_columns": ["total_charge"]},
        # an evaluation block with nothing to score
        {"labels": []},
        {"csv_name": "esol.csv"},  # unknown field (extra="forbid")
        {"geometry_origin": "xtb"},
        {"source_kind": ""},
        {"format_version": 2},
    ],
)
def test_spec_validators_refuse(update: dict) -> None:
    document = make_bundle_spec().model_dump(mode="json")
    with pytest.raises(pydantic.ValidationError):
        DatasetSpec.model_validate({**document, **update})


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


@pytest.mark.parametrize(
    "fields",
    [
        {"task_type": "multiclass"},  # no n_classes
        {"task_type": "multiclass", "n_classes": 1},
        {"task_type": "regression", "n_classes": 3},
        {"task_type": "regression", "name": ""},
        {"task_type": "multiclass", "n_classes": 3, "class_names": ["a", "b"]},
        {"task_type": "regression", "class_names": ["low", "high"]},
    ],
)
def test_label_column_validators_refuse(fields: dict) -> None:
    with pytest.raises(pydantic.ValidationError):
        LabelColumn.model_validate({"name": "label", **fields})


def test_task_set_maps_every_label_to_a_system_column() -> None:
    task_set = make_bundle_spec().task_set()
    assert [(config.name, config.task_type) for config in task_set.system_cols] == [
        ("activity", TaskType.classification),
        ("logp", TaskType.regression),
    ]
    assert all(config.scope is TaskScope.system for config in task_set.system_cols)
    assert task_set.atom_cols == []


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
    directory = tmp_path / "synthetic6"
    written = write_bundle(bundle, directory)

    # exactly the three files: no staging files are left behind
    assert sorted(path.name for path in directory.iterdir()) == THREE_FILES
    # the caller's provenance is not mutated by the writer
    assert bundle.provenance.outputs is None
    assert bundle.provenance.prepared_at is None

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
    # the data identity of a bundle is its table's content hash
    assert structures_identity(directory) == outputs.table_parquet.content_sha256


@pytest.mark.parametrize(
    "options",
    [{"smiles": False}, {"evaluation": False}, {"require_enantiomer_pairs": True}],
)
def test_every_bundle_variant_round_trips(tmp_path: Path, options: dict) -> None:
    reloaded = read_bundle(write_smiles_bundle(tmp_path, **options))
    assert list(reloaded.table.columns) == reloaded.spec.expected_columns()
    # a corpus (no evaluation block) counts no splits
    has_splits = bool(reloaded.provenance.counts.per_split)
    assert has_splits is (reloaded.spec.evaluation is not None)


def test_write_bundle_normalizes_dtypes(tmp_path: Path) -> None:
    """int32 ids and a plain-object split column are coerced."""
    bundle = make_bundle()
    bundle.table["stereoisomer_id"] = bundle.table["stereoisomer_id"].astype("int32")
    bundle.table["split"] = bundle.table["split"].astype(object)
    write_bundle(bundle, tmp_path / "synthetic6")
    reloaded = read_bundle(tmp_path / "synthetic6")
    assert reloaded.table["stereoisomer_id"].dtype == np.int64


def _give_the_spec_structures(bundle: Bundle) -> None:
    bundle.spec = bundle.spec.with_structures("etkdg_mmff")


def _leak_the_alanine_pair(bundle: Bundle) -> None:
    bundle.table["split"] = ["train", "valid", "test", "train", "test", "train"]


@pytest.mark.parametrize(
    "mutation, pattern",
    [(_give_the_spec_structures, ".*geometry_origin"), (_leak_the_alanine_pair, "6:")],
)
def test_write_bundle_refuses_and_writes_nothing(
    tmp_path: Path, mutation: Callable[[Bundle], None], pattern: str
) -> None:
    bundle = make_bundle()
    mutation(bundle)
    with pytest.raises(BundleValidationError) as raised:
        write_bundle(bundle, tmp_path / "broken")
    assert has_problem(raised.value.problems, pattern), raised.value.problems
    assert raised.value.directory == tmp_path / "broken"
    assert not (tmp_path / "broken").exists()


def _set_recorded_output(key: str, value: object) -> Callable[[dict], dict]:
    def edit(document: dict) -> dict:
        document["outputs"]["table.parquet"][key] = value
        return document

    return edit


def _without_outputs(document: dict) -> dict:
    return {key: value for key, value in document.items() if key != "outputs"}


def _set_format_version(version: object) -> Callable[[dict], dict]:
    return lambda document: {**document, "format_version": version}


TAMPERINGS: dict[str, tuple[str, Callable, str]] = {
    "tampered-hash": (
        PROVENANCE_FILENAME,
        _set_recorded_output("content_sha256", "0" * 64),
        ".*content_sha256",
    ),
    "wrong-row-count": (
        PROVENANCE_FILENAME,
        _set_recorded_output("rows", 7),
        ".*7 rows",
    ),
    "no-outputs": (PROVENANCE_FILENAME, _without_outputs, ".*no outputs"),
    "spec-not-a-mapping": (SPEC_FILENAME, lambda _: ["a", "list"], ".*not a mapping"),
    **{
        f"format-version-{version!r}": (
            SPEC_FILENAME,
            _set_format_version(version),
            "format_version",
        )
        for version in (0, 2, None, "1")
    },
    # a changed label value moves the content hash away from the record
    "tampered-table": (TABLE_FILENAME, set_value(0, "logp", 42.0), ".*content_sha256"),
    "invariant-on-disk": (TABLE_FILENAME, set_value(5, "split", "TEST"), "5:"),
}


@pytest.mark.parametrize(
    "filename, edit, pattern", TAMPERINGS.values(), ids=TAMPERINGS.keys()
)
def test_read_bundle_rejects_a_tampered_directory(
    tmp_path: Path, filename: str, edit: Callable, pattern: str
) -> None:
    path = write_smiles_bundle(tmp_path) / filename
    if filename == TABLE_FILENAME:
        table = pd.read_parquet(path)
        edit(table)
        table.to_parquet(path, index=False)
    else:
        path.write_text(yaml.safe_dump(edit(yaml.safe_load(path.read_text()))))
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(path.parent)
    assert has_problem(raised.value.problems, pattern), raised.value.problems


@pytest.mark.parametrize("missing", THREE_FILES)
def test_read_bundle_reports_a_missing_file(tmp_path: Path, missing: str) -> None:
    directory = write_smiles_bundle(tmp_path)
    (directory / missing).unlink()
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        read_bundle(directory)


def test_single_file_readers_do_not_validate(tmp_path: Path) -> None:
    directory = write_smiles_bundle(tmp_path)
    assert len(read_table(directory)) == 6
    subset = read_table(directory, columns=["stereoisomer_id", "split"])
    assert list(subset.columns) == ["stereoisomer_id", "split"]
    assert read_provenance(directory).dataset_id == "synthetic6"


# ------------------------------------------------------------ atomic writes


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


def test_a_dataset_records_its_structures_hash_as_its_identity(tmp_path: Path) -> None:
    spec, table = make_small_dataset()
    written = write_table_files(
        tmp_path, spec, table, make_provenance("small"), structures_sha256="a" * 64
    )
    assert written.outputs is not None
    assert written.outputs.structures_sha256 == "a" * 64
    assert sorted(path.name for path in tmp_path.iterdir()) == THREE_FILES
    assert read_provenance(tmp_path) == written
    assert structures_identity(tmp_path) == "a" * 64


def test_structures_identity_needs_a_provenance_with_outputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        structures_identity(tmp_path)
    (tmp_path / PROVENANCE_FILENAME).write_text(
        yaml.safe_dump(make_provenance("x").model_dump(mode="json", exclude_none=True))
    )
    with pytest.raises(BundleValidationError, match="no outputs"):
        structures_identity(tmp_path)


# ------------------------------------------------------- table invariants


def _reorder_columns(table: pd.DataFrame) -> None:
    table.insert(3, "logp", table.pop("logp"))


def _null_split_value(table: pd.DataFrame) -> None:
    table["split"] = table["split"].astype(object)
    table.loc[5, "split"] = None


BUNDLE_TABLE_MUTATIONS: dict[str, tuple[TableMutation, str]] = {
    "repeated-stereoisomer": (set_value(5, "stereoisomer_id", 0), "1:.*repeats"),
    "molecule-two-smiles": (set_value(5, "molecule_id", 0), "2:.*nonisomeric_smiles"),
    "asymmetric-enantiomer": (set_enantiomers(1), "4:.*not symmetric"),
    "reflexive-enantiomer": (set_enantiomers(0), "4:.*reflexive"),
    "dangling-enantiomer": (set_enantiomers(99), "4:.*absent"),
    "pair-across-molecules": (set_enantiomers(2, 0), "4:.*two molecule_ids"),
    "bad-split-value": (set_value(5, "split", "TEST"), "5:.*TEST"),
    "null-split-value": (_null_split_value, "5:.*<NA>"),
    "leak-in-split": (set_value(1, "split", "test"), "6:.*split "),
    "leak-in-second-split": (set_value(1, "split__random_s1", "valid"), "6:.*random"),
    "reordered-columns": (_reorder_columns, "7:.*columns"),
    "dropped-label": (drop("logp"), "7:.*columns"),
    "undeclared-column": (fill_integers("weight"), "7:.*columns"),
    "structure-id-on-a-bundle": (fill_integers("structure_id"), "7:.*columns"),
    "id-dtype": (retype("molecule_id", "int32"), "7:.*molecule_id"),
    "enantiomer-dtype": (retype("enantiomer_of", "float64"), "7:.*enantiomer_of"),
    "smiles-dtype": (fill_integers("isomeric_smiles"), "7:.*isomeric_smiles"),
    "split-dtype": (fill_integers("split__random_s1"), "7:.*split__random_s1"),
    "label-dtype": (fill_integers("activity"), "7:.*activity"),
    "classification-out-of-range": (set_value(0, "activity", 0.5), "7:.*0 or 1"),
}


@pytest.mark.parametrize(
    "mutation, pattern",
    BUNDLE_TABLE_MUTATIONS.values(),
    ids=BUNDLE_TABLE_MUTATIONS.keys(),
)
def test_bundle_table_invariants_are_detected(
    mutation: TableMutation, pattern: str
) -> None:
    spec, table = bundle_table()
    assert validate_table(spec, table) == []
    mutation(table)
    problems = validate_table(spec, table)
    assert has_problem(problems, pattern), problems


def _reverse_structure_ids(table: pd.DataFrame) -> None:
    table["structure_id"] = table["structure_id"].to_numpy()[::-1]


DATASET_TABLE_MUTATIONS: dict[str, tuple[TableMutation, str | None]] = {
    # several conformers of one stereoisomer are several dataset rows
    "repeated-stereoisomer-is-fine": (lambda table: None, None),
    "sparse-structure-id": (_reverse_structure_ids, "1:.*structure_id"),
    "stereoisomer-two-smiles": (
        set_value(3, "isomeric_smiles", "C[C@@H](N)C(=O)O"),
        "2:.*isomeric_smiles",
    ),
    "stereoisomer-two-molecules": (set_value(3, "molecule_id", 1), "3:"),
    "integer-charge": (retype("total_charge", "int64"), "7:.*total_charge"),
    "missing-multiplicity": (drop("multiplicity"), "7:.*columns"),
}


@pytest.mark.parametrize(
    "mutation, pattern",
    DATASET_TABLE_MUTATIONS.values(),
    ids=DATASET_TABLE_MUTATIONS.keys(),
)
def test_dataset_table_invariants_are_detected(
    mutation: TableMutation, pattern: str | None
) -> None:
    """Row 3 repeats stereoisomer 0, which only a dataset may do."""
    spec, table = make_small_dataset()
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    table["structure_id"] = np.arange(len(table), dtype="int64")
    mutation(table)
    problems = validate_table(spec, table)
    if pattern is None:
        assert problems == []
    else:
        assert has_problem(problems, pattern), problems


def test_require_enantiomer_pairs_rejects_a_lone_row() -> None:
    bundle = make_bundle()
    spec = make_bundle_spec(require_enantiomer_pairs=True)
    problems = validate_table(spec, normalize_table(bundle.table, spec))
    assert has_problem(problems, "4:.*require_enantiomer_pairs"), problems
    paired = make_bundle(require_enantiomer_pairs=True)
    assert validate_table(paired.spec, normalize_table(paired.table, spec)) == []


@pytest.mark.parametrize("bad_value", [3.0, -1.0, 1.5])
def test_multiclass_values_must_be_integers_in_range(bad_value: float) -> None:
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
    table = table.drop(columns=SMILES_COLUMNS)
    table["chirality_class"] = [0.0, 2.0, np.nan]
    assert validate_table(spec, normalize_table(table, spec)) == []
    table.loc[0, "chirality_class"] = bad_value
    assert has_problem(validate_table(spec, normalize_table(table, spec)), "7:")


def test_straddling_constitutions_are_counted_and_recorded(tmp_path: Path) -> None:
    bundle = make_bundle()
    assert count_stereoisomer_straddling_constitutions(bundle.table, "split") == 0
    with pytest.raises(KeyError):
        count_stereoisomer_straddling_constitutions(bundle.table, "split__absent")
    # The alanine pair is one constitution with two stereoisomers in two folds,
    # which a ``stereoisomer_id`` split group allows and the writer records.
    bundle.table.loc[1, "split"] = "test"
    assert count_stereoisomer_straddling_constitutions(bundle.table, "split") == 1
    assert bundle.spec.evaluation is not None
    bundle.spec = bundle.spec.model_copy(
        update={
            "evaluation": bundle.spec.evaluation.model_copy(
                update={"split_group": "stereoisomer_id"}
            )
        }
    )
    written = write_bundle(bundle, tmp_path / "synthetic6")
    assert written.counts.stereoisomer_straddling_constitutions == 1


# --------------------------------------------------- structure invariants


@pytest.fixture(scope="module")
def chiral_frames() -> list[Atoms]:
    return [embed_frame(smiles) for smiles in CHIRAL_DATASET_SMILES]


def _reflect_frame(index: int) -> Callable[[list[Atoms]], list[Atoms]]:
    def mutate(frames: list[Atoms]) -> list[Atoms]:
        reflect(frames[index])
        return frames

    return mutate


def _reverse_atom_order(frames: list[Atoms]) -> list[Atoms]:
    return [frames[0][::-1], *frames[1:]]


@pytest.mark.parametrize(
    "mutation, smiles, expected",
    [
        pytest.param(lambda frames: frames, True, None, id="matching"),
        pytest.param(
            lambda frames: frames[:2],
            True,
            "8: 2 structures for 3 table rows$",
            id="frame-count",
        ),
        pytest.param(_reflect_frame(1), True, r"9:.*rows \[1\]", id="reflected"),
        pytest.param(_reverse_atom_order, True, r"9:.*rows \[0\]", id="atom-order"),
        # invariant 9 does not run without SMILES to compare against
        pytest.param(_reflect_frame(1), False, None, id="reflected-without-smiles"),
    ],
)
def test_structure_invariants_8_and_9(
    chiral_frames: list[Atoms],
    mutation: Callable[[list[Atoms]], list[Atoms]],
    smiles: bool,
    expected: str | None,
) -> None:
    spec, table = make_small_dataset(smiles=smiles)
    frames = mutation([frame.copy() for frame in chiral_frames])
    problems = structure_problems(spec, table, frames, GEOMETRY_LIMITS)
    if expected is None:
        assert problems == []
    else:
        assert len(problems) == 1 and has_problem(problems, expected), problems


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
        (_methane_like("CO"), GeometryLimits(reject_zero_hydrogen=True), "no hydrogen"),
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
    assert len(problems) == 1 and has_problem(problems, f"10:.*{reason}"), problems


def test_many_offending_rows_are_summarised() -> None:
    spec, table = make_small_dataset(["CCO", "CCN", "CCC", "CO", "CN", "CS", "CF"])
    frames = [_methane_like("HH") for _ in range(len(table))]
    problems = structure_problems(spec, table, frames, GeometryLimits())
    assert has_problem(problems, r"10:.*7 rows, first \[0, 1, 2, 3, 4\]"), problems


# --- invariant 9 only compares what the SMILES actually specifies -----------


@pytest.mark.parametrize(
    "isomeric_smiles, reflected",
    [
        # One assigned and one unassigned tetrahedral centre: the embedding
        # picks some configuration for the second, which must not count.
        ("C[C@H](O)C(C)N", False),
        # Unspecified double bond next to an assigned centre: 3D perception
        # always yields E or Z, which must not count either.
        ("CC=C[C@H](C)O", False),
        # Unspecified imine.
        ("N=C(N)NC[C@@H]1COc2ccccc2O1", False),
        # An inverted *assigned* centre is still seen.
        ("C[C@H](O)C(C)N", True),
    ],
)
def test_invariant_9_compares_only_the_stereo_the_smiles_assigns(
    isomeric_smiles: str, reflected: bool
) -> None:
    frame = embed_frame(isomeric_smiles, seed=7)
    if reflected:
        reflect(frame)
    perceived = stereochemistry_from_frame(isomeric_smiles, frame)
    assert (perceived == tetrahedral_stereo_smiles(isomeric_smiles)) is not reflected


def test_stereochemistry_from_frame_returns_none_for_an_unusable_frame() -> None:
    assert stereochemistry_from_frame("not a smiles", Atoms("C")) is None
    assert stereochemistry_from_frame("C[C@H](N)C(=O)O", Atoms("C")) is None
    assert tetrahedral_stereo_smiles("not a smiles") is None


# ------------------------------------------------------------------ hashing


def test_parquet_file_hash_moves_but_content_hash_does_not(tmp_path: Path) -> None:
    table = make_bundle().table
    writer_settings: dict[str, tuple[Literal["snappy", "zstd"], int | None]] = {
        "a": ("snappy", None),
        "b": ("zstd", None),
        "c": ("snappy", 2),
        "d": ("snappy", None),
    }
    for name, (compression, row_group_size) in writer_settings.items():
        table.to_parquet(
            tmp_path / f"{name}.parquet",
            index=False,
            compression=compression,
            row_group_size=row_group_size,
        )
    digests = {
        name: hashlib.sha256((tmp_path / f"{name}.parquet").read_bytes()).hexdigest()
        for name in writer_settings
    }
    assert digests["a"] == digests["d"]
    assert digests["a"] != digests["b"]
    assert digests["a"] != digests["c"]
    reread = [pd.read_parquet(tmp_path / f"{name}.parquet") for name in writer_settings]
    assert len({content_hash_of_table(frame) for frame in reread}) == 1


@pytest.mark.parametrize(
    "mutation, moves",
    [
        (set_value(0, "logp", -2.9), True),
        (lambda table: table.rename(columns={"logp": "log_p"}, inplace=True), True),
        (retype("molecule_id", "int32"), True),
        # the string backend is not content
        (retype("isomeric_smiles", "object"), False),
        (retype("split", "object"), False),
    ],
)
def test_content_hash_reacts_to_values_names_and_dtypes(
    mutation: TableMutation, moves: bool
) -> None:
    table = make_bundle().table
    changed = table.copy()
    mutation(changed)
    assert (content_hash_of_table(changed) != content_hash_of_table(table)) is moves


def test_a_rewrite_of_the_same_bundle_records_the_same_content_hash(
    tmp_path: Path,
) -> None:
    first = write_bundle(make_bundle(), tmp_path / "a").outputs
    second = write_bundle(make_bundle(), tmp_path / "b").outputs
    assert first is not None and second is not None
    assert first.table_parquet.content_sha256 == second.table_parquet.content_sha256


# ---------------------------------------------------------------- discovery


def test_discovery_tells_bundles_from_datasets(tmp_path: Path) -> None:
    """A dataset is a bundle directory plus a ``dataset_config.yaml``."""
    write_smiles_bundle(tmp_path, dataset_id="zeta")
    write_smiles_bundle(tmp_path / "tdc", dataset_id="AMES")
    write_smiles_bundle(tmp_path, dataset_id="alpha")
    for root, dataset_id in [("", "omega"), ("", "beta"), ("nested", "too_deep")]:
        directory = write_smiles_bundle(tmp_path / root, dataset_id=dataset_id)
        (directory / DATASET_CONFIG_FILENAME).write_text("contains_smiles: false\n")
    (tmp_path / "config_only").mkdir()
    (tmp_path / "config_only" / DATASET_CONFIG_FILENAME).write_text("{}\n")
    (tmp_path / "status.yaml").write_text("n_tasks: 0\n")

    # bundles are found at any depth; datasets only directly under the root
    assert discover_bundles(tmp_path) == [
        tmp_path / "alpha",
        tmp_path / "tdc" / "AMES",
        tmp_path / "zeta",
    ]
    discovered = discover_datasets(tmp_path)
    assert [path.name for path, _ in discovered] == ["beta", "omega"]
    assert [spec.dataset_id for _, spec in discovered] == ["beta", "omega"]
    assert is_dataset_directory(tmp_path / "beta")
    assert not is_dataset_directory(tmp_path / "zeta")
    assert not is_dataset_directory(tmp_path / "config_only")
    assert discover_bundles(tmp_path / "missing") == []
    assert discover_datasets(tmp_path / "missing") == []


def test_discover_datasets_raises_on_an_unreadable_spec(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / SPEC_FILENAME).write_text("format_version: 99\n")
    (directory / DATASET_CONFIG_FILENAME).write_text("{}\n")
    with pytest.raises(BundleValidationError, match="format_version"):
        discover_datasets(tmp_path)


# --------------------------------------------------------------- preparer


def test_preparer_record_for_a_script_in_this_repository() -> None:
    record = PreparerRecord.for_script("molsuit/Rem3Di", Path(__file__))
    assert record.repo == "molsuit/Rem3Di"
    assert record.script.endswith("tests/test_bundle_format.py")
    assert record.git_sha is None or re.fullmatch(r"[0-9a-f]{40}", record.git_sha)


def test_preparer_record_path_is_relative_to_the_nearest_checkout(
    tmp_path: Path,
) -> None:
    """The nearest ``.git`` decides the relative path; an unreadable sha is None."""
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    script = repository / "preparers" / "tdc" / "prepare.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('prepare')\n")
    record = PreparerRecord.for_script("molsuit/remedi-data", script)
    assert record.script == "preparers/tdc/prepare.py"
    assert record.git_sha is None

    # outside any checkout the bare file name is recorded
    loose_script = tmp_path / "loose_script.py"
    loose_script.write_text("\n")
    record = PreparerRecord.for_script("somewhere", loose_script)
    if not any((parent / ".git").exists() for parent in loose_script.parents):
        assert record.script == "loose_script.py"
