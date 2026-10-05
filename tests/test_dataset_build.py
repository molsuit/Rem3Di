"""From a SMILES bundle to a dataset: expansion, the writer, the build run, verify.

``expand_to_structures`` does the row bookkeeping (dense ``structure_id``, id,
label and split carry-through, the orphan rule, charges); ``write_dataset``
validates and writes a zarr with the three table files beside it;
``build_datasets`` embeds conformers for every bundle, gates every frame and
writes and verifies each dataset; ``verify_dataset`` re-checks one on disk.

Embedding runs in-process (``n_workers=1``) over at most six small molecules,
so the monkeypatched ``embed_one_smiles`` and ``stereochemistry_from_frame``
are the ones actually called. The preparer -> build -> ``MoleculeDataset``
carry-through of ids, labels and splits lives in ``test_bundle_round_trip.py``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
import zarr
from ase import Atoms
from rdkit.Chem import AllChem, rdForceFieldHelpers

from remedi.data_handling.bundle import (
    CONFORMER_EMBEDDING_FAILED,
    ENANTIOMER_PARTNER_FAILED,
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    TABLE_FILENAME,
    Bundle,
    BundleValidationError,
    EvalMetric,
    ExpandedDataset,
    LabelColumn,
    canonical_smiles_pair,
    content_hash_of_table,
    expand_to_structures,
    read_bundle,
    stereochemistry_from_frame,
    structures_identity,
    validate_table,
    write_bundle,
)
from remedi.data_handling.chemistry.conformers import (
    ConformerEmbeddingConfig,
    charge_and_multiplicity,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import Split, TaskType
from remedi.data_handling.dataset_build import (
    DEFAULT_GEOMETRY_LIMITS,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    BuildSummary,
    DatasetBuildConfig,
    VerifyDatasetTask,
    VerifyReport,
    build_datasets,
    structures_sha256,
    verify_dataset,
    write_dataset,
    zarr_problems,
)
from remedi.data_handling.dataset_build import build as build_module
from remedi.evaluation.results import PydanticResult

from .helpers.bundle_fixtures import (
    ACHIRAL_TEN_SMILES,
    SMALL_ZARR_LAYOUT,
    embed_frame,
    make_bundle,
    make_dataset_spec,
    make_dataset_table,
    make_provenance,
    write_small_dataset,
    write_smiles_bundle,
)

DATASET_ID = "synthetic6"
L_ALANINE = "C[C@H](N)C(=O)O"
D_ALANINE = "C[C@@H](N)C(=O)O"
REGRESSION_LABEL = [LabelColumn(name="y", task_type=TaskType.regression)]


def canonical(smiles: str) -> str:
    return canonical_smiles_pair(smiles).isomeric


def stereoisomer_id_of(bundle: Bundle, smiles: str) -> int:
    rows = bundle.table[bundle.table["isomeric_smiles"] == canonical(smiles)]
    return int(rows["stereoisomer_id"].iloc[0])


def frames_for(
    bundle: Bundle,
    *,
    n_conformers: int = 1,
    failing: set[int] | None = None,
) -> dict[int, list[Atoms]]:
    """Seeded ETKDG frames per stereoisomer; ``failing`` ids get an empty list."""
    failing = failing or set()
    return {
        int(stereoisomer_id): (
            []
            if int(stereoisomer_id) in failing
            else [
                embed_frame(str(smiles), seed=seed + 1) for seed in range(n_conformers)
            ]
        )
        for stereoisomer_id, smiles in zip(
            bundle.table["stereoisomer_id"],
            bundle.table["isomeric_smiles"],
            strict=True,
        )
    }


def neutral_charges(bundle: Bundle) -> dict[int, tuple[float, float]]:
    return {
        int(stereoisomer_id): charge_and_multiplicity(str(smiles))
        for stereoisomer_id, smiles in zip(
            bundle.table["stereoisomer_id"],
            bundle.table["isomeric_smiles"],
            strict=True,
        )
    }


def expand(
    bundle: Bundle,
    frames: dict[int, list[Atoms]],
    charges: dict[int, tuple[float, float]] | None = None,
) -> ExpandedDataset:
    """ETKDG expansion; neutral charges unless ``charges`` is given."""
    return expand_to_structures(
        bundle,
        frames,
        neutral_charges(bundle) if charges is None else charges,
        geometry_origin="etkdg_mmff",
    )


def build_config(tmp_path: Path, **overrides: Any) -> DatasetBuildConfig:
    defaults: dict[str, Any] = dict(
        bundle_root=tmp_path / "bundles",
        zarr_root=tmp_path / "zarrs",
        n_workers=1,
        zarr=SMALL_ZARR_LAYOUT,
    )
    return DatasetBuildConfig(**{**defaults, **overrides})


def read_summary(config: DatasetBuildConfig, dataset_id: str) -> BuildSummary:
    document = yaml.safe_load(
        (Path(config.zarr_root) / "build" / f"{dataset_id}.yaml").read_text()
    )
    return BuildSummary.model_validate(document)


def write(expanded: ExpandedDataset, directory: Path, **overrides: Any):
    """``write_dataset`` of ``expanded``, with any argument replaced by ``overrides``."""
    arguments: dict[str, Any] = dict(
        spec=expanded.spec,
        table=expanded.table,
        structures=expanded.structures,
        provenance=expanded.provenance,
        directory=directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    return write_dataset(**{**arguments, **overrides})


def with_positions(
    structures: list[Atoms], index: int, transform: Callable[[np.ndarray], np.ndarray]
) -> list[Atoms]:
    """A copy of ``structures`` with frame ``index`` moved by ``transform``."""
    copied = [atoms.copy() for atoms in structures]
    copied[index].set_positions(transform(copied[index].get_positions()))
    return copied


@pytest.fixture(scope="module")
def expanded() -> ExpandedDataset:
    """The six-row bundle with one frame each; read-only, shared by the module."""
    bundle = make_bundle()
    return expand(bundle, frames_for(bundle))


# ------------------------------------------------------- expand_to_structures


def test_expansion_multiplies_rows_and_carries_everything_through() -> None:
    bundle = make_bundle()
    frames = frames_for(bundle, n_conformers=2)
    first = int(bundle.table.loc[0, "stereoisomer_id"])
    frames[first] = frames[first] + frames[first][:1]  # a variable count is fine
    expanded = expand(bundle, frames)

    table = expanded.table
    assert expanded.spec.geometry_origin == "etkdg_mmff"
    assert expanded.spec.has_structures
    assert list(table.columns) == expanded.spec.expected_columns()
    assert validate_table(expanded.spec, table) == []
    assert len(table) == len(expanded.structures) == 13
    assert table["structure_id"].tolist() == list(range(13))
    # every stereoisomer's conformers are adjacent, in bundle order
    assert table["stereoisomer_id"].tolist() == [
        identifier
        for identifier in bundle.table["stereoisomer_id"]
        for _ in range(3 if identifier == first else 2)
    ]
    carried = ["molecule_id", "enantiomer_of", "isomeric_smiles", "nonisomeric_smiles"]
    carried += ["activity", "logp", "split", "split__random_s1"]
    gathered = table.drop_duplicates("stereoisomer_id").set_index("stereoisomer_id")
    original = bundle.table.set_index("stereoisomer_id")
    pd.testing.assert_frame_equal(gathered[carried], original[carried])
    # each structure is a frame of its own row's molecule
    for atoms, smiles in zip(
        expanded.structures, table["isomeric_smiles"], strict=True
    ):
        assert stereochemistry_from_frame(smiles, atoms) is not None
    assert (table["total_charge"] == 0.0).all()
    assert (table["multiplicity"] == 1.0).all()
    assert expanded.dropped_counts == {}
    assert expanded.provenance.outputs is None
    assert expanded.provenance.counts.dropped == {"element_gate": 2}


@pytest.mark.parametrize("require_enantiomer_pairs", [False, True])
def test_the_orphan_rule(require_enantiomer_pairs: bool) -> None:
    """Without required pairs the partner is kept with ``enantiomer_of`` nulled;
    with them it is dropped too."""
    bundle = make_bundle(require_enantiomer_pairs=require_enantiomer_pairs)
    d_alanine = stereoisomer_id_of(bundle, D_ALANINE)
    l_alanine = stereoisomer_id_of(bundle, L_ALANINE)

    expanded = expand(bundle, frames_for(bundle, failing={d_alanine}))

    table = expanded.table
    remaining = set(table["stereoisomer_id"])
    assert d_alanine not in remaining
    if require_enantiomer_pairs:
        assert l_alanine not in remaining
        assert len(table) == 4
        assert table["enantiomer_of"].notna().all()
    else:
        assert len(table) == 5
        assert (
            table.loc[table["stereoisomer_id"] == l_alanine, "enantiomer_of"]
            .isna()
            .all()
        )
    expected = {CONFORMER_EMBEDDING_FAILED: 1, ENANTIOMER_PARTNER_FAILED: 1}
    assert expanded.dropped_counts == expected
    # merged onto the preparer's own counts
    assert expanded.provenance.counts.dropped == {"element_gate": 2, **expected}
    assert validate_table(expanded.spec, table) == []


def test_a_stereoisomer_absent_from_frames_and_charges_counts_as_failed() -> None:
    bundle = make_bundle()
    ethanol = stereoisomer_id_of(bundle, "CCO")
    frames, charges = frames_for(bundle), neutral_charges(bundle)
    del frames[ethanol], charges[ethanol]
    expanded = expand(bundle, frames, charges=charges)
    assert len(expanded.table) == 5
    assert expanded.dropped_counts == {CONFORMER_EMBEDDING_FAILED: 1}


def test_charges_are_taken_per_stereoisomer() -> None:
    bundle = make_bundle()
    charges = neutral_charges(bundle)
    ethanol = stereoisomer_id_of(bundle, "CCO")
    charges[ethanol] = (-1.0, 2.0)
    expanded = expand_to_structures(
        bundle, frames_for(bundle), charges, geometry_origin="source"
    )
    row = expanded.table[expanded.table["stereoisomer_id"] == ethanol]
    assert row["total_charge"].tolist() == [-1.0]
    assert row["multiplicity"].tolist() == [2.0]
    assert expanded.table["total_charge"].dtype == np.float64
    assert expanded.spec.geometry_origin == "source"

    del charges[ethanol]
    with pytest.raises(ValueError, match="total_charge"):
        expand(bundle, frames_for(bundle), charges=charges)


def test_expanding_a_dataset_is_refused() -> None:
    bundle = make_bundle()
    bundle.spec = bundle.spec.with_structures("etkdg_mmff")
    with pytest.raises(ValueError, match="already has structures"):
        expand_to_structures(bundle, {}, {}, geometry_origin="etkdg_mmff")


# --------------------------------------------------------------- write_dataset


def test_write_dataset_writes_a_zarr_and_the_three_files(
    tmp_path: Path, expanded: ExpandedDataset
) -> None:
    directory = tmp_path / DATASET_ID

    written = write(expanded, directory)

    names = {path.name for path in directory.iterdir()}
    assert {SPEC_FILENAME, TABLE_FILENAME, PROVENANCE_FILENAME} <= names
    assert {"dataset_config.yaml", "positions", "atomic_numbers", "ids"} <= names
    assert [path for path in tmp_path.iterdir() if path.suffix == ".partial"] == []
    reloaded = read_bundle(directory)
    assert reloaded.spec == expanded.spec
    assert reloaded.provenance == written
    assert written.counts.final_rows == 6
    assert written.counts.per_split == {
        "train": 3,
        "valid": 1,
        "test": 1,
        "unassigned": 1,
    }
    assert zarr_problems(directory, reloaded.spec, reloaded.table) == []

    assert written.outputs is not None
    assert structures_identity(directory) == written.outputs.structures_sha256
    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert written.outputs.structures_sha256 == structures_sha256(dataset)
        assert dataset.N_structures == 6
        assert dataset.config.contains_smiles is False
        assert dataset.config.tasks is not None
        assert [column.name for column in dataset.config.tasks.system_cols] == [
            "activity",
            "logp",
        ]
        for atoms, expected in zip(
            dataset.get_all_molecules(), expanded.structures, strict=True
        ):
            assert list(atoms.get_atomic_numbers()) == list(
                expected.get_atomic_numbers()
            )
            np.testing.assert_allclose(
                atoms.get_positions(), expected.get_positions(), atol=1e-5
            )
    finally:
        dataset.close()


def reflected(positions: np.ndarray) -> np.ndarray:
    return positions * [-1.0, 1.0, 1.0]


def second_atom_on_the_first(positions: np.ndarray) -> np.ndarray:
    return np.vstack([positions[:1], positions[:1], positions[2:]])


def invalid_arguments(expanded: ExpandedDataset, defect: str) -> dict[str, Any]:
    """``write_dataset`` arguments that break one invariant of ``expanded``."""
    structures, provenance = expanded.structures, expanded.provenance
    match defect:
        case "table":
            split = ["TEST", *expanded.table["split"].iloc[1:]]
            return {"table": expanded.table.assign(split=split)}
        case "count":
            return {"structures": structures[:-1]}
        case "reflected":
            return {"structures": with_positions(structures, 0, reflected)}
        case "overlap":
            return {
                "structures": with_positions(structures, 2, second_atom_on_the_first)
            }
        case "element":
            limits = provenance.geometry_limits.model_copy(
                update={"elements": ["C", "H"]}
            )
            return {
                "provenance": provenance.model_copy(update={"geometry_limits": limits})
            }
        case _:  # a bundle spec, one without structures
            return {"spec": expanded.spec.model_copy(update={"geometry_origin": None})}


#: Each defect and the invariant (or field) the refusal names.
EXPECTED_PROBLEM = {
    "table": "5:",
    "count": "8:",
    "reflected": "9:",
    "overlap": "10:",
    "element": "10:",
    "bundle_spec": "geometry_origin",
}


@pytest.mark.parametrize(("defect", "expected"), list(EXPECTED_PROBLEM.items()))
def test_write_dataset_refuses_an_invalid_dataset_and_leaves_nothing(
    tmp_path: Path, expanded: ExpandedDataset, defect: str, expected: str
) -> None:
    with pytest.raises(BundleValidationError) as raised:
        write(expanded, tmp_path / "out", **invalid_arguments(expanded, defect))
    problems = raised.value.problems
    assert any(expected in problem for problem in problems), problems
    assert list(tmp_path.iterdir()) == []


def test_a_failed_rewrite_keeps_the_previous_dataset(
    tmp_path: Path, expanded: ExpandedDataset
) -> None:
    directory = tmp_path / DATASET_ID
    first = write(expanded, directory)
    with pytest.raises(BundleValidationError):
        write(expanded, directory, structures=expanded.structures[:-1])
    assert read_bundle(directory).provenance == first
    assert verify_dataset(directory).problems == []


def test_a_successful_rewrite_replaces_the_dataset(
    tmp_path: Path, expanded: ExpandedDataset
) -> None:
    directory = tmp_path / DATASET_ID
    write(expanded, directory)
    (directory / "stale_file.txt").write_text("from the previous build")
    bundle = make_bundle()
    frames = frames_for(bundle)
    del frames[stereoisomer_id_of(bundle, "CCO")]

    write(expand(bundle, frames), directory)

    assert not (directory / "stale_file.txt").exists()
    report = verify_dataset(directory)
    assert report.problems == []
    assert (report.dataset_id, report.rows) == (DATASET_ID, 5)
    assert report.structures_sha256 == structures_identity(directory)


def test_structures_sha256_moves_with_the_coordinates(
    tmp_path: Path, expanded: ExpandedDataset
) -> None:
    moved = with_positions(expanded.structures, 3, lambda positions: positions + 0.01)
    outputs = [
        write(expanded, tmp_path / name, structures=structures).outputs
        for name, structures in (("a", expanded.structures), ("b", moved))
    ]
    assert outputs[0] is not None and outputs[1] is not None
    assert outputs[0].structures_sha256 != outputs[1].structures_sha256
    # the table did not change, so neither did its content hash
    assert (
        outputs[0].table_parquet.content_sha256
        == outputs[1].table_parquet.content_sha256
    )


def test_zarr_problems_compares_the_zarr_to_the_table_row_for_row(
    tmp_path: Path, expanded: ExpandedDataset
) -> None:
    directory = tmp_path / DATASET_ID
    write(expanded, directory)
    reloaded = read_bundle(directory)
    spec, table = reloaded.spec, reloaded.table

    assert zarr_problems(directory, spec, table.iloc[:-1]) == [
        "8: the zarr holds 6 structures for 5 rows"
    ]
    changes: list[tuple[str, Any, str]] = [
        ("molecule_id", table["molecule_id"][::-1].to_numpy(), "ids/molecule_id"),
        ("split", "test", "tasks/split does not match the table's default split"),
        ("logp", 99.0, "the zarr's labels do not match the table"),
        ("activity", np.nan, "the zarr's labels do not match the table"),
    ]
    for column, value, expected in changes:
        changed = table.copy()
        if column == "molecule_id":
            changed[column] = value
        else:
            changed.loc[0, column] = value
        (problem,) = zarr_problems(directory, spec, changed)
        assert problem.startswith("8:") and expected in problem


def test_a_labelled_corpus_without_evaluation_is_written_unassigned(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(evaluation=False)
    directory = tmp_path / "corpus"
    write(expand(bundle, frames_for(bundle)), directory)
    assert read_bundle(directory).spec.evaluation is None
    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert set(np.asarray(dataset.split[:]).tolist()) == {Split.unassigned.value}
    finally:
        dataset.close()
    assert verify_dataset(directory).problems == []


def test_a_corpus_without_labels_or_evaluation_is_written(tmp_path: Path) -> None:
    directory = write_small_dataset(
        tmp_path, dataset_id="corpus", labels=[], metrics=None, targets=None
    )
    reloaded = read_bundle(directory)
    assert reloaded.spec.labels == [] and reloaded.spec.evaluation is None
    assert verify_dataset(directory).problems == []


def test_a_source_dataset_without_smiles_is_written(tmp_path: Path) -> None:
    """A structure-shipping source (tmQM-like) with source geometry and no SMILES."""
    smiles_values = ACHIRAL_TEN_SMILES[:4]
    labels = [LabelColumn(name="gap", task_type=TaskType.regression)]
    spec_arguments: dict[str, Any] = dict(
        dataset_id="source_like", labels=labels, metrics=[EvalMetric.mae]
    )
    spec = make_dataset_spec(**spec_arguments, extra_columns=["csd_code"]).model_copy(
        update={"smiles": False, "geometry_origin": "source"}
    )
    table = make_dataset_table(
        make_dataset_spec(**spec_arguments),
        smiles_values,
        np.arange(4, dtype=float),
        {"split": ["train", "train", "valid", "test"]},
    ).drop(columns=["isomeric_smiles", "nonisomeric_smiles"])
    table["csd_code"] = ["AAAA", "BBBB", "CCCC", "DDDD"]
    directory = tmp_path / "source_like"
    write_dataset(
        spec,
        table,
        [embed_frame(smiles) for smiles in smiles_values],
        make_provenance("source_like"),
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    reloaded = read_bundle(directory)
    assert reloaded.table["csd_code"].tolist() == ["AAAA", "BBBB", "CCCC", "DDDD"]
    assert verify_dataset(directory).problems == []


# -------------------------------------------------------------- build_datasets


def write_charged_bundle(directory: Path) -> None:
    """:func:`make_bundle` with ethanol replaced by a cation."""
    bundle = make_bundle(dataset_id=directory.name)
    table = bundle.table.copy()
    for column in ("isomeric_smiles", "nonisomeric_smiles"):
        table[column] = table[column].astype(object)
        table.loc[table["isomeric_smiles"] == "CCO", column] = "NCC[NH3+]"
    bundle.table = table
    write_bundle(bundle, directory)


def test_build_datasets_builds_and_verifies_every_bundle(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    write_charged_bundle(tmp_path / "bundles" / "second")
    config = build_config(tmp_path)

    report = build_datasets(config)

    assert report.n_failed == 0
    assert [(status.name, status.ok) for status in report.statuses] == [
        ("0_build_dataset_second", True),
        ("1_build_dataset_synthetic6", True),
        ("2_verify_dataset_second", True),
        ("3_verify_dataset_synthetic6", True),
    ]
    zarr_root = tmp_path / "zarrs"
    status = yaml.safe_load((zarr_root / "status.yaml").read_text())
    assert status["n_tasks"] == 4 and status["n_failed"] == 0
    assert (
        DatasetBuildConfig.model_validate(
            yaml.safe_load((zarr_root / "manifest.yaml").read_text())
        )
        == config
    )

    directory = zarr_root / DATASET_ID
    dataset = read_bundle(directory)
    assert dataset.spec.geometry_origin == "etkdg_mmff"
    assert len(dataset.table) == 6
    assert len((directory / TIMINGS_FILENAME).read_text().splitlines()) == 6

    bundle = read_bundle(tmp_path / "bundles" / DATASET_ID)
    record = dataset.provenance.conformers
    assert record is not None
    assert record.settings == config.conformers
    assert record.parent_bundle_content_sha256 == content_hash_of_table(bundle.table)
    assert dataset.provenance.geometry_limits == config.geometry_limits
    assert dataset.provenance.counts.dropped == {
        "element_gate": 2,
        CONFORMER_EMBEDDING_FAILED: 0,
        STEREO_MISMATCH_AFTER_EMBEDDING: 0,
        ENANTIOMER_PARTNER_FAILED: 0,
    }
    summary = read_summary(config, DATASET_ID)
    assert not summary.skipped
    assert (summary.stereoisomers_in, summary.structures_out) == (6, 6)
    assert summary.bundle_content_sha256 == content_hash_of_table(bundle.table)
    assert dataset.provenance.outputs is not None
    assert (
        summary.table_content_sha256
        == dataset.provenance.outputs.table_parquet.content_sha256
    )
    report_document = yaml.safe_load(
        (zarr_root / "verify" / f"{DATASET_ID}.yaml").read_text()
    )
    assert VerifyReport.model_validate(report_document).problems == []

    # the charge is taken from each SMILES and carried into the zarr
    charged = read_bundle(zarr_root / "second").table
    assert charged["total_charge"].tolist() == [
        1.0 if smiles == "NCC[NH3+]" else 0.0 for smiles in charged["isomeric_smiles"]
    ]
    molecule_dataset = MoleculeDataset.open_existing_dataset_from_dir(
        zarr_root / "second"
    )
    try:
        total_charge = np.asarray(molecule_dataset.total_charge[:]).tolist()
    finally:
        molecule_dataset.close()
    assert total_charge == charged["total_charge"].tolist()
    assert 1.0 in total_charge


@pytest.fixture(scope="module")
def built_once(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A bundle root and the zarr root built from it; copy before changing."""
    root = tmp_path_factory.mktemp("built_once")
    write_smiles_bundle(root / "bundles", dataset_id=DATASET_ID)
    assert build_datasets(build_config(root)).n_failed == 0
    return root


@pytest.mark.parametrize(
    ("overrides", "rewrite_bundle", "rows"),
    [
        pytest.param({}, False, None, id="unchanged-skips"),
        pytest.param(
            {"conformers": ConformerEmbeddingConfig(n_conformers=2)},
            False,
            12,
            id="conformer-settings",
        ),
        # alanine (13 atoms), phenol (13) and ethanol (9) fit; tartaric acid
        # (16) and 2-butanol (15) are dropped
        pytest.param(
            {
                "geometry_limits": DEFAULT_GEOMETRY_LIMITS.model_copy(
                    update={"max_atoms": 13}
                )
            },
            False,
            4,
            id="geometry-limits",
        ),
        pytest.param({}, True, 6, id="changed-bundle"),
        pytest.param({"overwrite": True}, False, 6, id="overwrite"),
    ],
)
def test_a_rerun_rebuilds_only_when_something_changed(
    tmp_path: Path,
    built_once: Path,
    overrides: dict[str, Any],
    rewrite_bundle: bool,
    rows: int | None,
) -> None:
    """``rows=None`` means the rerun must skip and leave the dataset untouched."""
    shutil.copytree(built_once, tmp_path, dirs_exist_ok=True)
    table_path = tmp_path / "zarrs" / DATASET_ID / TABLE_FILENAME
    before = table_path.stat().st_mtime_ns
    if rewrite_bundle:
        write_smiles_bundle(
            tmp_path / "bundles", dataset_id=DATASET_ID, require_enantiomer_pairs=True
        )
    config = build_config(tmp_path, **overrides)

    assert build_datasets(config).n_failed == 0

    summary = read_summary(config, DATASET_ID)
    if rows is None:
        assert summary.skipped and summary.structures_out == 0
        assert table_path.stat().st_mtime_ns == before
        return
    assert not summary.skipped and summary.structures_out == rows
    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert len(dataset.table) == rows
    assert dataset.provenance.conformers is not None
    assert dataset.provenance.conformers.settings == config.conformers
    assert dataset.provenance.geometry_limits == config.geometry_limits
    too_large = 2 if "geometry_limits" in overrides else 0
    assert dataset.provenance.counts.dropped.get("max_atoms", 0) == too_large
    assert dataset.spec.evaluation is not None
    assert dataset.spec.evaluation.require_enantiomer_pairs == rewrite_bundle


def test_dataset_ids_select_nested_bundles_by_their_relative_path(
    tmp_path: Path,
) -> None:
    write_smiles_bundle(tmp_path / "bundles" / "tdc", dataset_id="AMES")
    write_smiles_bundle(tmp_path / "bundles", dataset_id="ignored")
    config = build_config(tmp_path, dataset_ids=["tdc/AMES"], verify=False)
    assert build_config(tmp_path).resolve_dataset_ids() == ["ignored", "tdc/AMES"]

    report = build_datasets(config)

    assert report.n_failed == 0
    assert [status.name for status in report.statuses] == ["0_build_dataset_tdc__AMES"]
    assert read_bundle(tmp_path / "zarrs" / "tdc" / "AMES").spec.dataset_id == "AMES"
    assert not (tmp_path / "zarrs" / "ignored").exists()


@pytest.mark.parametrize("keep_going", [True, False])
def test_a_failing_bundle_and_keep_going(tmp_path: Path, keep_going: bool) -> None:
    """With ``keep_going`` the other bundles are still built; without it the run
    raises at the first failure. Either way the status records the failure."""
    write_smiles_bundle(
        tmp_path / "bundles", dataset_id="a_without_smiles", smiles=False
    )
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path, keep_going=keep_going)

    if keep_going:
        report = build_datasets(config)
        assert report.n_failed == 1
        failed, built, verified = report.statuses
        assert not failed.ok and "smiles: false" in (failed.error or "")
        assert built.ok and verified.ok
        assert verified.name.endswith(DATASET_ID)
    else:
        with pytest.raises(ValueError, match="smiles: false"):
            build_datasets(config)
    assert (tmp_path / "zarrs" / DATASET_ID).exists() == keep_going
    assert not (tmp_path / "zarrs" / "a_without_smiles").exists()
    status = yaml.safe_load((tmp_path / "zarrs" / "status.yaml").read_text())
    assert status["n_failed"] == 1


@pytest.mark.parametrize("tampered", [False, True])
def test_verify_covers_datasets_a_preparer_wrote_directly(
    tmp_path: Path, tampered: bool
) -> None:
    directory = write_small_dataset(
        tmp_path / "zarrs",
        dataset_id="direct",
        labels=REGRESSION_LABEL,
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 1.0, 10),
    )
    if tampered:
        tamper_provenance_structures_hash(directory)
    (tmp_path / "bundles").mkdir()

    report = build_datasets(build_config(tmp_path))

    assert report.n_failed == int(tampered)
    (status,) = report.statuses
    assert status.name == "0_verify_dataset_direct"
    assert ("structures_sha256" in (status.error or "")) == tampered
    assert (tmp_path / "zarrs" / "verify" / "direct.yaml").is_file()


def test_a_stereo_mismatch_after_embedding_is_dropped_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MMFF walking a centre through planarity is injected at the perception step."""
    target = canonical(D_ALANINE)

    def flipped_stereochemistry(isomeric_smiles: str, atoms: Atoms) -> str | None:
        if canonical(isomeric_smiles) == target:
            return canonical(L_ALANINE)
        return stereochemistry_from_frame(isomeric_smiles, atoms)

    monkeypatch.setattr(
        build_module, "stereochemistry_from_frame", flipped_stereochemistry
    )
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path)

    assert build_datasets(config).n_failed == 0

    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert len(dataset.table) == 5
    dropped = dataset.provenance.counts.dropped
    assert dropped[STEREO_MISMATCH_AFTER_EMBEDDING] == 1
    # rejected for its stereochemistry, not miscounted as a failed embedding
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 0
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 1
    assert read_summary(config, DATASET_ID).dropped == dropped


#: What RDKit's BFGS line search throws on a pathological molecule: a bare
#: RuntimeError out of the C++ layer.
INVARIANT_VIOLATION = (
    "Invariant Violation\n\tbad direction in linearSearch\n\t"
    "Violation occurred on line 100 in file Numerics/Optimizer/BFGSOpt.h"
)


def test_one_raising_molecule_does_not_sink_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_mmff = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs
    calls = {"count": 0}

    def raise_on_the_first_molecule(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError(INVARIANT_VIOLATION)
        return real_mmff(*args, **kwargs)

    monkeypatch.setattr(
        AllChem, "MMFFOptimizeMoleculeConfs", raise_on_the_first_molecule
    )
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)

    assert build_datasets(build_config(tmp_path)).n_failed == 0

    directory = tmp_path / "zarrs" / DATASET_ID
    dataset = read_bundle(directory)
    assert dataset.provenance.counts.dropped[CONFORMER_EMBEDDING_FAILED] == 1
    # L-alanine is first: its partner D-alanine survives as an orphan
    assert dataset.provenance.counts.dropped[ENANTIOMER_PARTNER_FAILED] == 1
    assert len(dataset.table) == 5
    statuses = [
        yaml.safe_load(line)["status"]
        for line in (directory / TIMINGS_FILENAME).read_text().splitlines()
    ]
    assert statuses.count("mmff_failed") == 1


# -------------------------------------------------------------- verify_dataset


@pytest.fixture
def ten_row_dataset(tmp_path: Path) -> Path:
    """Ten rows, so the small zarr layout spreads them over several batches and shards."""
    return write_small_dataset(
        tmp_path,
        dataset_id="ten",
        labels=REGRESSION_LABEL,
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 1.0, 10),
    )


def tamper_provenance_structures_hash(directory: Path) -> None:
    provenance_path = directory / PROVENANCE_FILENAME
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["structures_sha256"] = "0" * 64
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))


def overlap_two_atoms(directory: Path) -> None:
    positions = zarr.open_array(directory / "positions", mode="r+")
    shifted = np.asarray(positions[:])
    shifted[1] = shifted[0]  # two atoms of the first molecule on top of each other
    positions[:] = shifted


def reverse_molecule_ids(directory: Path) -> None:
    molecule_ids = zarr.open_array(directory / "ids" / "molecule_id", mode="r+")
    molecule_ids[:] = np.asarray(molecule_ids[:])[::-1]


def change_a_label_in_the_table(directory: Path) -> None:
    table = pd.read_parquet(directory / TABLE_FILENAME)
    table.loc[0, "y"] = 42.0
    table.to_parquet(directory / TABLE_FILENAME, index=False)


def delete_the_table(directory: Path) -> None:
    (directory / TABLE_FILENAME).unlink()


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        (overlap_two_atoms, ["structures_sha256", "10:"]),
        (reverse_molecule_ids, ["8: ids/molecule_id does not match the table"]),
        (change_a_label_in_the_table, ["content_sha256"]),
        (tamper_provenance_structures_hash, ["structures_sha256"]),
        (delete_the_table, [TABLE_FILENAME]),
    ],
)
def test_verify_catches_tampering(
    ten_row_dataset: Path, tamper: Callable[[Path], None], expected: list[str]
) -> None:
    """Every fragment shows up in its own problem, and nothing else is reported;
    a missing file is reported rather than raised."""
    tamper(ten_row_dataset)
    problems = verify_dataset(ten_row_dataset).problems
    assert len(problems) == len(expected), problems
    for fragment in expected:
        assert any(fragment in problem for problem in problems), problems


def test_the_verify_task_yields_its_report_then_raises(ten_row_dataset: Path) -> None:
    tamper_provenance_structures_hash(ten_row_dataset)
    results = VerifyDatasetTask(dataset_path=ten_row_dataset).run(None)

    first = next(results)
    assert isinstance(first, PydanticResult)
    assert isinstance(first.obj, VerifyReport)
    with pytest.raises(ValueError, match="failed verification"):
        next(results)
