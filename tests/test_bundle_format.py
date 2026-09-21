"""Tests for the prepared-benchmark-bundle format (``BENCHMARK_DATA_FORMAT.md`` §1).

Run without the repository conftest (which imports torch); this package is
deliberately torch-free::

    uv run --no-sync pytest --noconftest tests/test_bundle_format.py -q
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from rdkit import Chem
from rdkit.Chem import AllChem

from remedi.data_handling.bundle import (
    CONFORMER_EMBEDDING_FAILED,
    ENANTIOMER_PARTNER_FAILED,
    BenchmarkSpec,
    BenchmarkTask,
    Bundle,
    BundleCounts,
    BundleProvenance,
    BundleValidationError,
    ConformerGenerationRecord,
    EtkdgParameters,
    EvalMetric,
    GeometryLimits,
    MmffParameters,
    PreparerRecord,
    SourceRecord,
    assign_identity,
    content_hash_of_table,
    count_stereoisomer_straddling_constitutions,
    discover_bundles,
    expand_to_conformers,
    mirror_isomeric_smiles,
    read_bundle,
    read_table,
    stereochemistry_from_frame,
    tetrahedral_stereo_smiles,
    validate_bundle,
    write_bundle,
)
from remedi.data_handling.dataset.tasks import TaskScope, TaskType

# Six rows: an enantiomer pair (0, 1), an achiral molecule (2), a meso compound (3),
# a duplicate spelling of the achiral molecule (4) and a lone chiral molecule whose
# mirror image is absent (5).
SIX_SMILES = [
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine
    "c1ccccc1O",  # phenol
    "O[C@@H]([C@@H](O)C(O)=O)C(O)=O",  # meso-tartaric acid
    "Oc1ccccc1",  # phenol again — same stereoisomer
    "C[C@H](O)CC",  # (S)-2-butanol, partner absent
]

GEOMETRY_LIMITS = GeometryLimits(
    max_atoms=200,
    elements=["H", "C", "N", "O"],
    reject_zero_hydrogen=True,
    min_hydrogen_heavy_ratio=0.1,
    min_interatomic_distance=0.5,
)


# --------------------------------------------------------------------- fixtures


def make_spec(
    *,
    stage: str = "smiles",
    split_columns: list[str] | None = None,
    default_split: str = "split",
    split_group: str = "molecule_id",
    require_enantiomer_pairs: bool = False,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        dataset_id="synthetic6",
        description="Six synthetic rows exercising every identity case of §1.1.",
        tasks=[
            BenchmarkTask(name="activity", task_type=TaskType.classification),
            BenchmarkTask(name="logp", task_type=TaskType.regression),
        ],
        metrics=["AUROC", "RMSE"],
        stage=stage,
        geometry_origin="etkdg_mmff" if stage == "conformers" else None,
        split_columns=split_columns or ["split", "split__random_s1"],
        default_split=default_split,
        split_group=split_group,
        require_enantiomer_pairs=require_enantiomer_pairs,
        source_kind="synthetic",
    )


def make_provenance(dataset_id: str = "synthetic6") -> BundleProvenance:
    return BundleProvenance(
        dataset_id=dataset_id,
        preparer=PreparerRecord(
            repo="molsuit/remedi-data", script="tests/synthetic.py"
        ),
        source=SourceRecord(geometry="synthetic ETKDG embeddings"),
        geometry_limits=GEOMETRY_LIMITS,
        counts=BundleCounts(source_molecules=len(SIX_SMILES)),
        notices=[],
    )


def make_six_row_bundle(spec: BenchmarkSpec | None = None) -> Bundle:
    spec = spec or make_spec()
    table = assign_identity(SIX_SMILES).to_frame()
    table["activity"] = np.array([1.0, 0.0, np.nan, 1.0, np.nan, 0.0])
    table["logp"] = np.array([-3.0, -3.0, 1.5, np.nan, 1.5, 0.6])
    table["split"] = table["molecule_id"].map(
        {0: "train", 1: "valid", 2: "test", 3: "unassigned"}
    )
    table["split__random_s1"] = table["molecule_id"].map(
        {0: "test", 1: "train", 2: "train", 3: "valid"}
    )
    return Bundle(
        spec=spec,
        table=table[spec.expected_columns()],
        structures=None,
        provenance=make_provenance(),
    )


def embed_stereoisomer(isomeric_smiles: str, seed: int = 1) -> Atoms:
    """One ETKDG frame in RDKit ``AddHs`` atom order — what invariant 9 assumes."""
    molecule = Chem.AddHs(Chem.MolFromSmiles(isomeric_smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    assert AllChem.EmbedMolecule(molecule, parameters) == 0, isomeric_smiles
    return Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        positions=molecule.GetConformer().GetPositions(),
    )


def make_conformers_bundle() -> Bundle:
    """The same six rows, one real ETKDG frame each."""
    smiles_bundle = make_six_row_bundle()
    table = smiles_bundle.table.copy()
    structures = []
    for row_index, isomeric_smiles in enumerate(table["isomeric_smiles"]):
        atoms = embed_stereoisomer(str(isomeric_smiles))
        atoms.info["structure_id"] = row_index
        structures.append(atoms)
    return Bundle(
        spec=make_spec(stage="conformers"),
        table=table,
        structures=structures,
        provenance=make_provenance(),
    )


# ------------------------------------------------------------------ round trips


def test_identity_assignment_of_the_six_rows() -> None:
    table = make_six_row_bundle().table
    assert table.loc[0, "molecule_id"] == table.loc[1, "molecule_id"]
    assert table.loc[0, "stereoisomer_id"] != table.loc[1, "stereoisomer_id"]
    assert table.loc[0, "enantiomer_of"] == table.loc[1, "stereoisomer_id"]
    assert table.loc[1, "enantiomer_of"] == table.loc[0, "stereoisomer_id"]
    # the duplicate spelling of phenol collapses onto one stereoisomer
    assert table.loc[2, "stereoisomer_id"] == table.loc[4, "stereoisomer_id"]
    # achiral, meso and partner-less rows all have no mirror partner in the bundle
    assert table.loc[[2, 3, 4, 5], "enantiomer_of"].isna().all()
    assert list(table.columns) == make_spec().expected_columns()


def test_smiles_bundle_round_trips_through_disk(tmp_path: Path) -> None:
    bundle = make_six_row_bundle()
    assert validate_bundle(bundle) == []
    directory = write_bundle(bundle, tmp_path / "synthetic6")
    assert sorted(path.name for path in directory.iterdir()) == [
        "benchmark.yaml",
        "provenance.yaml",
        "table.parquet",
    ]
    reloaded = read_bundle(directory)
    pd.testing.assert_frame_equal(reloaded.table, bundle.table)
    assert reloaded.spec == bundle.spec
    assert reloaded.structures is None
    assert str(reloaded.table["enantiomer_of"].dtype) == "Int64"
    assert reloaded.table["activity"].isna().sum() == 2

    counts = reloaded.provenance.counts
    assert counts.final_rows == 6
    assert counts.per_split == {"train": 2, "valid": 2, "test": 1, "unassigned": 1}
    assert counts.per_task_non_null == {"activity": 4, "logp": 5}
    assert counts.stereoisomer_straddling_constitutions == 0
    assert counts.source_molecules == 6
    outputs = reloaded.provenance.outputs
    assert outputs is not None
    assert outputs.table_parquet.rows == 6
    assert outputs.table_parquet.content_sha256 == content_hash_of_table(bundle.table)
    assert outputs.structures_extxyz is None
    assert reloaded.provenance.prepared_at is not None


def test_conformers_bundle_round_trips_through_disk(tmp_path: Path) -> None:
    bundle = make_conformers_bundle()
    assert validate_bundle(bundle) == []
    directory = write_bundle(bundle, tmp_path / "synthetic6_3d")
    assert (directory / "structures.extxyz").is_file()
    reloaded = read_bundle(directory)
    assert reloaded.structures is not None
    assert len(reloaded.structures) == 6
    for index, (before, after) in enumerate(
        zip(bundle.structures or [], reloaded.structures, strict=True)
    ):
        assert after.info["structure_id"] == index
        assert np.abs(after.get_positions() - before.get_positions()).max() < 1e-7
    outputs = reloaded.provenance.outputs
    assert outputs is not None
    assert outputs.structures_extxyz is not None
    assert outputs.structures_extxyz.frames == 6


def test_discover_bundles_finds_and_sorts(tmp_path: Path) -> None:
    for dataset_id in ("zeta", "alpha"):
        bundle = make_six_row_bundle()
        bundle.spec = bundle.spec.model_copy(update={"dataset_id": dataset_id})
        write_bundle(bundle, tmp_path / dataset_id)
    (tmp_path / "not_a_bundle").mkdir()
    assert discover_bundles(tmp_path) == [tmp_path / "alpha", tmp_path / "zeta"]
    assert discover_bundles(tmp_path / "missing") == []


# ------------------------------------------------------------------- invariants


def _reverse_structure_id(table: pd.DataFrame) -> None:
    table["structure_id"] = table["structure_id"].to_numpy()[::-1]


def _split_two_smiles_onto_one_stereoisomer(table: pd.DataFrame) -> None:
    values = list(table["isomeric_smiles"])
    values[4] = "CCO"
    table["isomeric_smiles"] = pd.array(values, dtype="str")


def _split_two_nonisomeric_onto_one_molecule(table: pd.DataFrame) -> None:
    values = list(table["nonisomeric_smiles"])
    values[4] = "CCO"
    table["nonisomeric_smiles"] = pd.array(values, dtype="str")


def _make_stereoisomer_span_two_molecules(table: pd.DataFrame) -> None:
    values = list(table["stereoisomer_id"])
    values[4] = 0
    table["stereoisomer_id"] = np.array(values, dtype="int64")


def _break_enantiomer_symmetry(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array(
        [1, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int64"
    )


def _make_enantiomer_reflexive(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array([0, 1, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int64")


def _dangling_enantiomer_pointer(table: pd.DataFrame) -> None:
    table["enantiomer_of"] = pd.array(
        [99, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int64"
    )


def _bad_split_value(table: pd.DataFrame) -> None:
    table["split"] = pd.array(
        ["train", "train", "TEST", "unassigned", "TEST", "train"], dtype="str"
    )


def _leak_across_split_group(table: pd.DataFrame) -> None:
    table["split"] = pd.array(
        ["train", "valid", "test", "unassigned", "test", "train"], dtype="str"
    )


def _wrong_task_dtype(table: pd.DataFrame) -> None:
    table["activity"] = table["activity"].fillna(0).astype("int64")


def _drop_a_task_column(table: pd.DataFrame) -> None:
    table.drop(columns=["logp"], inplace=True)


@pytest.mark.parametrize(
    "mutation, expected_prefix",
    [
        (_reverse_structure_id, "1:"),
        (_split_two_smiles_onto_one_stereoisomer, "2:"),
        (_split_two_nonisomeric_onto_one_molecule, "2:"),
        (_make_stereoisomer_span_two_molecules, "3:"),
        (_break_enantiomer_symmetry, "4:"),
        (_make_enantiomer_reflexive, "4:"),
        (_dangling_enantiomer_pointer, "4:"),
        (_bad_split_value, "5:"),
        (_leak_across_split_group, "6:"),
        (_wrong_task_dtype, "7:"),
        (_drop_a_task_column, "7:"),
    ],
)
def test_table_invariants_are_detected(
    mutation: Callable[[pd.DataFrame], None], expected_prefix: str
) -> None:
    bundle = make_six_row_bundle()
    mutation(bundle.table)
    problems = validate_bundle(bundle)
    assert any(problem.startswith(expected_prefix) for problem in problems), problems


def test_multiclass_values_must_lie_in_range() -> None:
    spec = BenchmarkSpec(
        dataset_id="chirality",
        tasks=[
            BenchmarkTask(
                name="chirality_class", task_type=TaskType.multiclass, n_classes=3
            )
        ],
        metrics=["balanced-accuracy"],
        stage="smiles",
        split_columns=["split"],
        default_split="split",
        split_group="molecule_id",
        source_kind="synthetic",
    )
    bundle = make_six_row_bundle()
    table = bundle.table[list(bundle.table.columns[:6])].copy()
    table["chirality_class"] = np.array([0.0, 1.0, 2.0, np.nan, 2.0, 1.0])
    table["split"] = pd.array(
        ["train", "train", "test", "valid", "test", "train"], dtype="str"
    )
    good = Bundle(spec=spec, table=table, structures=None, provenance=make_provenance())
    assert validate_bundle(good) == []
    table.loc[0, "chirality_class"] = 3.0
    assert any(problem.startswith("7:") for problem in validate_bundle(good))


def test_missing_frames_and_mislabelled_frames_fail_invariant_8() -> None:
    bundle = make_conformers_bundle()
    assert bundle.structures is not None
    short = Bundle(
        spec=bundle.spec,
        table=bundle.table,
        structures=bundle.structures[:5],
        provenance=bundle.provenance,
    )
    assert any(problem.startswith("8:") for problem in validate_bundle(short))

    bundle.structures[2].info["structure_id"] = 99
    assert any(problem.startswith("8:") for problem in validate_bundle(bundle))

    without = Bundle(
        spec=make_spec(stage="conformers"),
        table=make_six_row_bundle().table,
        structures=None,
        provenance=make_provenance(),
    )
    assert any(problem.startswith("8:") for problem in validate_bundle(without))


def test_reflected_geometry_fails_invariant_9() -> None:
    bundle = make_conformers_bundle()
    assert bundle.structures is not None
    reflected = bundle.structures[0].get_positions() * np.array([-1.0, 1.0, 1.0])
    bundle.structures[0].set_positions(reflected)
    problems = validate_bundle(bundle)
    assert any(problem.startswith("9:") for problem in problems), problems


def test_overlapping_atoms_fail_invariant_10() -> None:
    bundle = make_conformers_bundle()
    assert bundle.structures is not None
    positions = bundle.structures[2].get_positions()
    positions[1] = positions[0]
    bundle.structures[2].set_positions(positions)
    problems = validate_bundle(bundle)
    assert any(problem.startswith("10:") for problem in problems), problems


def test_element_gate_and_size_gate_fail_invariant_10() -> None:
    bundle = make_conformers_bundle()
    bundle.provenance.geometry_limits = GeometryLimits(
        max_atoms=3, elements=["H", "C"], min_interatomic_distance=0.5
    )
    problems = validate_bundle(bundle)
    assert sum(problem.startswith("10:") for problem in problems) >= 2, problems


def test_require_enantiomer_pairs_rejects_a_lone_row() -> None:
    bundle = make_six_row_bundle(make_spec(require_enantiomer_pairs=True))
    problems = validate_bundle(bundle)
    assert any(
        problem.startswith("4:") and "require_enantiomer_pairs" in problem
        for problem in problems
    ), problems


def test_write_bundle_refuses_an_invalid_bundle(tmp_path: Path) -> None:
    bundle = make_six_row_bundle()
    _leak_across_split_group(bundle.table)
    with pytest.raises(BundleValidationError) as raised:
        write_bundle(bundle, tmp_path / "broken")
    assert any(problem.startswith("6:") for problem in raised.value.problems)
    assert not (tmp_path / "broken" / "table.parquet").exists()


def test_straddling_constitutions_are_counted() -> None:
    bundle = make_six_row_bundle(make_spec(split_group="stereoisomer_id"))
    assert count_stereoisomer_straddling_constitutions(bundle.table, "split") == 0
    _leak_across_split_group(bundle.table)
    # The alanine pair is one constitution with two stereoisomers in two folds.
    assert count_stereoisomer_straddling_constitutions(bundle.table, "split") == 1
    assert validate_bundle(bundle) == []


# ------------------------------------------------------------------------ spec


def test_spec_validators() -> None:
    with pytest.raises(ValueError):
        make_spec(split_columns=["split__a"])  # default_split "split" is absent
    with pytest.raises(ValueError):
        make_spec(split_columns=["split", "folds"], default_split="split")
    with pytest.raises(ValueError):
        BenchmarkSpec(
            **{**make_spec().model_dump(), "geometry_origin": "source"}
        )  # smiles stage must leave geometry_origin unset
    with pytest.raises(ValueError):
        BenchmarkSpec(
            **{
                **make_spec().model_dump(),
                "stage": "conformers",
                "geometry_origin": None,
            }
        )
    with pytest.raises(ValueError):  # duplicate task names
        BenchmarkSpec(
            **{
                **make_spec().model_dump(),
                "tasks": [
                    {"name": "activity", "task_type": "classification"},
                    {"name": "activity", "task_type": "regression"},
                ],
            }
        )
    with pytest.raises(ValueError):  # a task colliding with a fixed column
        BenchmarkSpec(
            **{
                **make_spec().model_dump(),
                "tasks": [{"name": "molecule_id", "task_type": "regression"}],
            }
        )
    with pytest.raises(ValueError):  # an extra column colliding with a task
        BenchmarkSpec(**{**make_spec().model_dump(), "extra_columns": ["logp"]})
    with pytest.raises(ValueError):  # multiclass without n_classes
        BenchmarkTask(name="chirality_class", task_type=TaskType.multiclass)
    with pytest.raises(ValueError):  # n_classes below 2
        BenchmarkTask(
            name="chirality_class", task_type=TaskType.multiclass, n_classes=1
        )
    with pytest.raises(ValueError):  # n_classes on a non-multiclass task
        BenchmarkTask(name="logp", task_type=TaskType.regression, n_classes=3)
    with pytest.raises(ValueError):  # extra="forbid"
        BenchmarkSpec(**{**make_spec().model_dump(), "csv_name": "esol.csv"})


def test_expected_columns_is_the_declared_order() -> None:
    spec = BenchmarkSpec(
        dataset_id="ordered",
        tasks=[BenchmarkTask(name="rs", task_type=TaskType.classification)],
        metrics=["AUROC"],
        stage="smiles",
        split_columns=["split", "split__scaffold_s0"],
        default_split="split__scaffold_s0",
        split_group="molecule_id",
        extra_columns=["n_chiral", "total_charge"],
        source_kind="remedi_data_csv",
    )
    assert spec.expected_columns() == [
        "structure_id",
        "stereoisomer_id",
        "molecule_id",
        "isomeric_smiles",
        "nonisomeric_smiles",
        "enantiomer_of",
        "rs",
        "split",
        "split__scaffold_s0",
        "n_chiral",
        "total_charge",
    ]


# --------------------------------------------------------------------- hashing


def test_parquet_file_hash_moves_but_content_hash_does_not(tmp_path: Path) -> None:
    table = make_six_row_bundle().table
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


def test_content_hash_reacts_to_a_value_change() -> None:
    table = make_six_row_bundle().table
    before = content_hash_of_table(table)
    changed = table.copy()
    changed.loc[0, "logp"] = -2.9
    assert content_hash_of_table(changed) != before
    renamed = table.rename(columns={"logp": "log_p"})
    assert content_hash_of_table(renamed) != before


def test_extxyz_round_trip_keeps_precision_and_structure_id(tmp_path: Path) -> None:
    generator = np.random.default_rng(0)
    positions = generator.normal(size=(5, 3)) * 3.0
    atoms = Atoms("CCCCO", positions=positions)
    atoms.info["structure_id"] = 0
    ase_write(tmp_path / "structures.extxyz", [atoms], format="extxyz")
    reread = ase_read(tmp_path / "structures.extxyz", index=":")[0]
    assert reread.info["structure_id"] == 0
    assert isinstance(reread.info["structure_id"], int | np.integer)
    assert np.abs(reread.get_positions() - positions).max() < 1e-7


# ------------------------------------------------------------ tampered on disk


def test_read_bundle_rejects_a_tampered_content_hash(tmp_path: Path) -> None:
    directory = write_bundle(make_six_row_bundle(), tmp_path / "synthetic6")
    provenance_path = directory / "provenance.yaml"
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["table.parquet"]["content_sha256"] = "0" * 64
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert any("content_sha256" in problem for problem in raised.value.problems)


def test_read_bundle_rejects_a_tampered_extxyz_hash(tmp_path: Path) -> None:
    directory = write_bundle(make_conformers_bundle(), tmp_path / "synthetic6_3d")
    provenance_path = directory / "provenance.yaml"
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["structures.extxyz"]["file_sha256"] = "0" * 64
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert any("structures.extxyz" in problem for problem in raised.value.problems)


def test_read_bundle_rejects_another_format_version(tmp_path: Path) -> None:
    directory = write_bundle(make_six_row_bundle(), tmp_path / "synthetic6")
    spec_path = directory / "benchmark.yaml"
    document = yaml.safe_load(spec_path.read_text())
    document["format_version"] = 2
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(BundleValidationError) as raised:
        read_bundle(directory)
    assert any("format_version" in problem for problem in raised.value.problems)


def test_read_bundle_reports_a_missing_file(tmp_path: Path) -> None:
    directory = write_bundle(make_six_row_bundle(), tmp_path / "synthetic6")
    (directory / "table.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        read_bundle(directory)


# ------------------------------------------------------------ stage transition


def structures_for(
    bundle: Bundle, *, n_conformers: int, failing_stereoisomer_ids: set[int]
) -> dict[int, list[Atoms]]:
    frames: dict[int, list[Atoms]] = {}
    for stereoisomer_id, isomeric_smiles in zip(
        bundle.table["stereoisomer_id"],
        bundle.table["isomeric_smiles"],
        strict=True,
    ):
        stereoisomer_id = int(stereoisomer_id)
        if stereoisomer_id in failing_stereoisomer_ids:
            frames[stereoisomer_id] = []
            continue
        frames[stereoisomer_id] = [
            embed_stereoisomer(str(isomeric_smiles), seed=seed + 1)
            for seed in range(n_conformers)
        ]
    return frames


def make_unique_stereoisomer_bundle(
    *, require_enantiomer_pairs: bool = False
) -> Bundle:
    """The six rows deduplicated to one row per stereoisomer (smiles-stage shape)."""
    bundle = make_six_row_bundle(
        make_spec(require_enantiomer_pairs=require_enantiomer_pairs)
    )
    table = bundle.table.drop_duplicates("stereoisomer_id").reset_index(drop=True)
    table["structure_id"] = np.arange(len(table), dtype="int64")
    bundle.table = table
    return bundle


def test_expand_to_conformers_keeps_identity_and_applies_the_orphan_rule(
    tmp_path: Path,
) -> None:
    bundle = make_unique_stereoisomer_bundle()
    d_alanine_id = int(bundle.table.loc[1, "stereoisomer_id"])
    l_alanine_id = int(bundle.table.loc[0, "stereoisomer_id"])
    expanded = expand_to_conformers(
        bundle,
        structures_for(bundle, n_conformers=2, failing_stereoisomer_ids={d_alanine_id}),
        conformer_record=ConformerGenerationRecord(
            rdkit_version="test",
            n_conformers_requested=2,
            etkdg=EtkdgParameters(max_iterations=200),
            mmff=MmffParameters(max_iterations=100, non_bonded_threshold=100.0),
        ),
    )
    result = expanded.bundle
    assert result.spec.stage == "conformers"
    assert result.spec.geometry_origin == "etkdg_mmff"
    assert validate_bundle(result) == []
    # 5 stereoisomers in, 1 failed to embed, 4 survivors x 2 conformers
    assert len(result.table) == 8
    assert np.array_equal(result.table["structure_id"], np.arange(8))
    assert result.structures is not None
    assert [atoms.info["structure_id"] for atoms in result.structures] == list(range(8))

    for column in (
        "molecule_id",
        "isomeric_smiles",
        "nonisomeric_smiles",
        "split",
        "activity",
    ):
        gathered = result.table.drop_duplicates("stereoisomer_id").set_index(
            "stereoisomer_id"
        )[column]
        original = bundle.table.set_index("stereoisomer_id")[column]
        pd.testing.assert_series_equal(
            gathered, original.loc[gathered.index], check_names=False
        )

    # L-alanine's partner failed: the pointer is nullified, the row survives.
    orphan_rows = result.table[result.table["stereoisomer_id"] == l_alanine_id]
    assert len(orphan_rows) == 2
    assert orphan_rows["enantiomer_of"].isna().all()
    assert expanded.dropped_counts == {
        CONFORMER_EMBEDDING_FAILED: 1,
        ENANTIOMER_PARTNER_FAILED: 1,
    }
    assert result.provenance.counts.dropped == expanded.dropped_counts
    assert result.provenance.conformers is not None
    assert result.provenance.conformers.parent_bundle_content_sha256 == (
        content_hash_of_table(bundle.table)
    )

    directory = write_bundle(result, tmp_path / "expanded")
    reloaded = read_bundle(directory)
    assert reloaded.structures is not None
    assert len(reloaded.structures) == 8
    assert reloaded.provenance.counts.final_rows == 8


def test_expand_to_conformers_drops_the_orphan_when_pairs_are_required() -> None:
    paired = make_unique_stereoisomer_bundle(require_enantiomer_pairs=True)
    # Keep only the enantiomer pair, which is the only part that can satisfy
    # require_enantiomer_pairs at the smiles stage.
    paired.table = paired.table.iloc[:2].reset_index(drop=True)
    paired.table["structure_id"] = np.arange(2, dtype="int64")
    assert validate_bundle(paired) == []

    d_alanine_id = int(paired.table.loc[1, "stereoisomer_id"])
    expanded = expand_to_conformers(
        paired,
        structures_for(paired, n_conformers=1, failing_stereoisomer_ids={d_alanine_id}),
    )
    assert len(expanded.bundle.table) == 0
    assert expanded.dropped_counts == {
        CONFORMER_EMBEDDING_FAILED: 1,
        ENANTIOMER_PARTNER_FAILED: 1,
    }
    assert validate_bundle(expanded.bundle) == []


def test_expand_to_conformers_refuses_a_conformers_stage_input() -> None:
    with pytest.raises(ValueError):
        expand_to_conformers(make_conformers_bundle(), {})


def test_expand_to_conformers_refuses_repeated_stereoisomers() -> None:
    with pytest.raises(ValueError):
        expand_to_conformers(make_six_row_bundle(), {})


# ------------------------------------------------------- chemistry edge cases


def test_meso_compound_has_no_enantiomer() -> None:
    meso = Chem.MolToSmiles(Chem.MolFromSmiles("O[C@@H]([C@@H](O)C(O)=O)C(O)=O"))
    assert mirror_isomeric_smiles(meso) == meso
    identity = assign_identity([meso])
    assert identity.enantiomer_of.isna().all()


def test_two_stereocentres_pair_only_exact_mirror_images() -> None:
    identity = assign_identity(
        [
            "C[C@H](O)[C@H](C)N",
            "C[C@@H](O)[C@@H](C)N",
            "C[C@H](O)[C@@H](C)N",
            "C[C@@H](O)[C@H](C)N",
        ]
    )
    frame = identity.to_frame()
    assert frame["molecule_id"].nunique() == 1
    assert frame["stereoisomer_id"].nunique() == 4
    assert frame["enantiomer_of"].notna().all()
    assert frame.loc[0, "enantiomer_of"] == frame.loc[1, "stereoisomer_id"]
    assert frame.loc[2, "enantiomer_of"] == frame.loc[3, "stereoisomer_id"]


# ------------------------------------------------------ consumer-side helpers


def test_eval_metric_members_match_the_registry_they_replace() -> None:
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


def test_task_set_maps_every_task_to_a_system_column() -> None:
    task_set = make_spec().task_set()
    assert [config.name for config in task_set.system_cols] == ["activity", "logp"]
    assert [config.task_type for config in task_set.system_cols] == [
        TaskType.classification,
        TaskType.regression,
    ]
    assert all(config.scope is TaskScope.system for config in task_set.system_cols)
    assert task_set.atom_cols == []
    assert task_set.system_map == {"activity": 0, "logp": 1}


def test_read_table_reads_columns_without_validating(tmp_path: Path) -> None:
    directory = write_bundle(make_six_row_bundle(), tmp_path / "synthetic6")
    full = read_table(directory)
    assert len(full) == 6
    subset = read_table(directory, columns=["stereoisomer_id", "split"])
    assert list(subset.columns) == ["stereoisomer_id", "split"]
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nowhere")


# --- invariant 9 only compares what the SMILES actually specifies -----------


def _embedded_frame(isomeric_smiles: str) -> Atoms:
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles(isomeric_smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 7
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    return Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        positions=molecule.GetConformer().GetPositions(),
    )


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
    frame = _embedded_frame(isomeric_smiles)
    assert stereochemistry_from_frame(isomeric_smiles, frame) == (
        tetrahedral_stereo_smiles(isomeric_smiles)
    )


def test_invariant_9_still_sees_an_inverted_assigned_centre() -> None:
    isomeric_smiles = "C[C@H](O)C(C)N"
    frame = _embedded_frame(isomeric_smiles)
    reflected = frame.copy()
    reflected.set_positions(frame.get_positions() * np.array([-1.0, 1.0, 1.0]))
    assert stereochemistry_from_frame(isomeric_smiles, reflected) != (
        tetrahedral_stereo_smiles(isomeric_smiles)
    )
