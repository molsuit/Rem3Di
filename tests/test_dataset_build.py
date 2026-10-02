"""From a SMILES bundle to a dataset: expansion, the writer, the build run, verify.

``expand_to_structures`` does the row bookkeeping (dense ``structure_id``, id,
label and split carry-through, the orphan rule, charges); ``write_dataset``
validates and writes a zarr with the three table files beside it;
``build_datasets`` embeds conformers for every bundle, gates every frame and
writes and verifies each dataset; ``verify_dataset`` re-checks one on disk.

Embedding runs in-process (``n_workers=1``) over at most six small molecules,
so the monkeypatched ``embed_one_smiles`` and ``stereochemistry_from_frame``
are the ones actually called.
"""

from __future__ import annotations

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
    LabelColumn,
    canonical_smiles_pair,
    content_hash_of_table,
    expand_to_structures,
    read_bundle,
    stereochemistry_from_frame,
    structures_identity,
    validate_table,
)
from remedi.data_handling.chemistry import conformers as conformers_module
from remedi.data_handling.chemistry.conformers import (
    ConformerEmbeddingConfig,
    ConformerTimingRecord,
    EmbedResult,
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
    TEN_ROW_SPLIT,
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
    frames: dict[int, list[Atoms]] = {}
    for stereoisomer_id, smiles in zip(
        bundle.table["stereoisomer_id"], bundle.table["isomeric_smiles"], strict=True
    ):
        frames[int(stereoisomer_id)] = (
            []
            if int(stereoisomer_id) in failing
            else [
                embed_frame(str(smiles), seed=seed + 1) for seed in range(n_conformers)
            ]
        )
    return frames


def neutral_charges(bundle: Bundle) -> dict[int, tuple[float, float]]:
    return {
        int(stereoisomer_id): charge_and_multiplicity(str(smiles))
        for stereoisomer_id, smiles in zip(
            bundle.table["stereoisomer_id"],
            bundle.table["isomeric_smiles"],
            strict=True,
        )
    }


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


def leftover_staging(root: Path) -> list[Path]:
    return [path for path in root.iterdir() if path.name.endswith(".partial")]


# ------------------------------------------------------- expand_to_structures


def test_expansion_multiplies_rows_and_carries_everything_through() -> None:
    bundle = make_bundle()
    expanded = expand_to_structures(
        bundle,
        frames_for(bundle, n_conformers=2),
        neutral_charges(bundle),
        geometry_origin="etkdg_mmff",
    )

    table = expanded.table
    assert expanded.spec.geometry_origin == "etkdg_mmff"
    assert expanded.spec.has_structures
    assert list(table.columns) == expanded.spec.expected_columns()
    assert validate_table(expanded.spec, table) == []
    assert len(table) == len(expanded.structures) == 12
    assert table["structure_id"].tolist() == list(range(12))
    # every stereoisomer's two conformers are adjacent, in bundle order
    assert table["stereoisomer_id"].tolist() == [
        identifier for identifier in bundle.table["stereoisomer_id"] for _ in range(2)
    ]
    carried = [
        "molecule_id",
        "enantiomer_of",
        "isomeric_smiles",
        "nonisomeric_smiles",
        "activity",
        "logp",
        "split",
        "split__random_s1",
    ]
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


def test_expansion_keeps_a_variable_number_of_frames_per_stereoisomer() -> None:
    bundle = make_bundle()
    frames = frames_for(bundle)
    first = int(bundle.table.loc[0, "stereoisomer_id"])
    frames[first] = frames[first] * 3
    expanded = expand_to_structures(
        bundle, frames, neutral_charges(bundle), geometry_origin="etkdg_mmff"
    )
    assert len(expanded.table) == 8
    assert expanded.table["stereoisomer_id"].tolist()[:4] == [first, first, first, 1]


def test_the_orphan_rule_nulls_the_surviving_partner() -> None:
    bundle = make_bundle()
    d_alanine = stereoisomer_id_of(bundle, D_ALANINE)
    l_alanine = stereoisomer_id_of(bundle, L_ALANINE)

    expanded = expand_to_structures(
        bundle,
        frames_for(bundle, failing={d_alanine}),
        neutral_charges(bundle),
        geometry_origin="etkdg_mmff",
    )

    table = expanded.table
    assert len(table) == 5
    assert d_alanine not in set(table["stereoisomer_id"])
    orphan = table[table["stereoisomer_id"] == l_alanine]
    assert len(orphan) == 1 and orphan["enantiomer_of"].isna().all()
    assert expanded.dropped_counts == {
        CONFORMER_EMBEDDING_FAILED: 1,
        ENANTIOMER_PARTNER_FAILED: 1,
    }
    # merged onto the preparer's own counts
    assert expanded.provenance.counts.dropped == {
        "element_gate": 2,
        CONFORMER_EMBEDDING_FAILED: 1,
        ENANTIOMER_PARTNER_FAILED: 1,
    }
    assert validate_table(expanded.spec, table) == []


def test_the_orphan_rule_drops_the_partner_when_pairs_are_required() -> None:
    bundle = make_bundle(require_enantiomer_pairs=True)
    d_alanine = stereoisomer_id_of(bundle, D_ALANINE)
    l_alanine = stereoisomer_id_of(bundle, L_ALANINE)

    expanded = expand_to_structures(
        bundle,
        frames_for(bundle, failing={d_alanine}),
        neutral_charges(bundle),
        geometry_origin="etkdg_mmff",
    )

    remaining = set(expanded.table["stereoisomer_id"])
    assert remaining.isdisjoint({d_alanine, l_alanine})
    assert len(expanded.table) == 4
    assert expanded.table["enantiomer_of"].notna().all()
    assert expanded.dropped_counts == {
        CONFORMER_EMBEDDING_FAILED: 1,
        ENANTIOMER_PARTNER_FAILED: 1,
    }
    assert validate_table(expanded.spec, expanded.table) == []


def test_a_stereoisomer_absent_from_the_frames_counts_as_failed() -> None:
    bundle = make_bundle()
    frames = frames_for(bundle)
    del frames[stereoisomer_id_of(bundle, "CCO")]
    expanded = expand_to_structures(
        bundle, frames, neutral_charges(bundle), geometry_origin="etkdg_mmff"
    )
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


def test_a_missing_charge_for_a_survivor_is_refused() -> None:
    bundle = make_bundle()
    charges = neutral_charges(bundle)
    del charges[stereoisomer_id_of(bundle, "CCO")]
    with pytest.raises(ValueError, match="total_charge"):
        expand_to_structures(
            bundle, frames_for(bundle), charges, geometry_origin="etkdg_mmff"
        )


def test_a_missing_charge_for_a_failed_stereoisomer_is_fine() -> None:
    bundle = make_bundle()
    ethanol = stereoisomer_id_of(bundle, "CCO")
    charges = neutral_charges(bundle)
    del charges[ethanol]
    expanded = expand_to_structures(
        bundle,
        frames_for(bundle, failing={ethanol}),
        charges,
        geometry_origin="etkdg_mmff",
    )
    assert len(expanded.table) == 5


def test_expanding_a_dataset_is_refused() -> None:
    bundle = make_bundle()
    bundle.spec = bundle.spec.with_structures("etkdg_mmff")
    with pytest.raises(ValueError, match="already has structures"):
        expand_to_structures(bundle, {}, {}, geometry_origin="etkdg_mmff")


# --------------------------------------------------------------- write_dataset


def expanded_dataset(**bundle_options):
    bundle = make_bundle(**bundle_options)
    return expand_to_structures(
        bundle,
        frames_for(bundle),
        neutral_charges(bundle),
        geometry_origin="etkdg_mmff",
    )


def test_write_dataset_writes_a_zarr_and_the_three_files(tmp_path: Path) -> None:
    expanded = expanded_dataset()
    directory = tmp_path / DATASET_ID

    written = write_dataset(
        expanded.spec,
        expanded.table,
        expanded.structures,
        expanded.provenance,
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )

    names = {path.name for path in directory.iterdir()}
    assert {SPEC_FILENAME, TABLE_FILENAME, PROVENANCE_FILENAME} <= names
    assert {"dataset_config.yaml", "positions", "atomic_numbers", "ids"} <= names
    assert leftover_staging(tmp_path) == []
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
    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert written.outputs.structures_sha256 == structures_sha256(dataset)
        assert dataset.N_structures == 6
        assert dataset.config.contains_smiles is False
    finally:
        dataset.close()
    assert structures_identity(directory) == written.outputs.structures_sha256


def test_the_zarr_holds_the_tables_ids_labels_masks_and_split_codes(
    tmp_path: Path,
) -> None:
    expanded = expanded_dataset()
    directory = tmp_path / DATASET_ID
    write_dataset(
        expanded.spec,
        expanded.table,
        expanded.structures,
        expanded.provenance,
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    table = read_bundle(directory).table

    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert np.array_equal(dataset.structure_ids[:], table["structure_id"])
        assert np.array_equal(dataset.molecule_ids[:], table["molecule_id"])
        assert np.array_equal(dataset.isomer_ids[:], table["stereoisomer_id"])
        labels = table[["activity", "logp"]].to_numpy()
        masks = np.asarray(dataset.mask_system[:])
        assert np.array_equal(masks.astype(bool), ~np.isnan(labels))
        targets = np.asarray(dataset.targets_system[:], dtype=np.float64)
        present = ~np.isnan(labels)
        np.testing.assert_allclose(targets[present], labels[present], rtol=1e-6)
        expected_codes = [
            {
                "train": Split.train.value,
                "valid": Split.valid.value,
                "test": Split.test.value,
                "unassigned": Split.unassigned.value,
            }[name]
            for name in table["split"]
        ]
        assert np.asarray(dataset.split[:]).tolist() == expected_codes
        assert dataset.config.tasks is not None
        assert [column.name for column in dataset.config.tasks.system_cols] == [
            "activity",
            "logp",
        ]
        np.testing.assert_allclose(dataset.total_charge[:], table["total_charge"])
        np.testing.assert_allclose(dataset.multiplicity[:], table["multiplicity"])
        frames = dataset.get_all_molecules()
        for atoms, expected in zip(frames, expanded.structures, strict=True):
            assert list(atoms.get_atomic_numbers()) == list(
                expected.get_atomic_numbers()
            )
            np.testing.assert_allclose(
                atoms.get_positions(), expected.get_positions(), atol=1e-5
            )
        assert dataset.get_smiles_per_structure() == table["isomeric_smiles"].tolist()
    finally:
        dataset.close()


def test_write_dataset_writes_across_several_batches_and_shards(
    tmp_path: Path,
) -> None:
    directory = write_small_dataset(
        tmp_path,
        dataset_id="ten",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 1.0, 10),
    )
    report = verify_dataset(directory)
    assert report.problems == []
    assert report.rows == 10


def test_write_dataset_refuses_an_invalid_table_and_leaves_nothing(
    tmp_path: Path,
) -> None:
    expanded = expanded_dataset()
    table = expanded.table.copy()
    table.loc[0, "split"] = "TEST"
    with pytest.raises(BundleValidationError) as raised:
        write_dataset(
            expanded.spec,
            table,
            expanded.structures,
            expanded.provenance,
            tmp_path / DATASET_ID,
        )
    assert any(problem.startswith("5:") for problem in raised.value.problems)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("defect", ["count", "reflected", "overlap", "element"])
def test_write_dataset_refuses_invalid_structures_and_leaves_nothing(
    tmp_path: Path, defect: str
) -> None:
    expanded = expanded_dataset()
    structures = [atoms.copy() for atoms in expanded.structures]
    provenance = expanded.provenance
    expected_prefix = {"count": "8:", "reflected": "9:", "overlap": "10:"}.get(
        defect, "10:"
    )
    if defect == "count":
        structures = structures[:-1]
    elif defect == "reflected":
        structures[0].set_positions(
            structures[0].get_positions() * np.array([-1.0, 1.0, 1.0])
        )
    elif defect == "overlap":
        positions = structures[2].get_positions()
        positions[1] = positions[0]
        structures[2].set_positions(positions)
    else:
        provenance = provenance.model_copy(deep=True)
        provenance.geometry_limits = provenance.geometry_limits.model_copy(
            update={"elements": ["C", "H"]}
        )
    with pytest.raises(BundleValidationError) as raised:
        write_dataset(
            expanded.spec, expanded.table, structures, provenance, tmp_path / "out"
        )
    assert any(
        problem.startswith(expected_prefix) for problem in raised.value.problems
    ), raised.value.problems
    assert list(tmp_path.iterdir()) == []


def test_write_dataset_refuses_a_bundle_spec(tmp_path: Path) -> None:
    expanded = expanded_dataset()
    bundle_spec = expanded.spec.model_copy(update={"geometry_origin": None})
    with pytest.raises(BundleValidationError, match="geometry_origin"):
        write_dataset(
            bundle_spec,
            expanded.table,
            expanded.structures,
            expanded.provenance,
            tmp_path / "out",
        )
    assert list(tmp_path.iterdir()) == []


def test_a_failed_rewrite_keeps_the_previous_dataset(tmp_path: Path) -> None:
    expanded = expanded_dataset()
    directory = tmp_path / DATASET_ID
    first = write_dataset(
        expanded.spec,
        expanded.table,
        expanded.structures,
        expanded.provenance,
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    with pytest.raises(BundleValidationError):
        write_dataset(
            expanded.spec,
            expanded.table,
            expanded.structures[:-1],
            expanded.provenance,
            directory,
        )
    assert read_bundle(directory).provenance == first
    assert verify_dataset(directory).problems == []


def test_a_successful_rewrite_replaces_the_dataset(tmp_path: Path) -> None:
    expanded = expanded_dataset()
    directory = tmp_path / DATASET_ID
    arguments = (expanded.spec, expanded.table, expanded.structures)
    write_dataset(*arguments, expanded.provenance, directory, layout=SMALL_ZARR_LAYOUT)
    (directory / "stale_file.txt").write_text("from the previous build")
    shorter = expand_to_structures(
        make_bundle(),
        {
            key: frames
            for key, frames in frames_for(make_bundle()).items()
            if key != stereoisomer_id_of(make_bundle(), "CCO")
        },
        neutral_charges(make_bundle()),
        geometry_origin="etkdg_mmff",
    )
    write_dataset(
        shorter.spec,
        shorter.table,
        shorter.structures,
        shorter.provenance,
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    assert len(read_bundle(directory).table) == 5
    assert not (directory / "stale_file.txt").exists()
    assert verify_dataset(directory).problems == []


def test_structures_sha256_moves_with_the_coordinates(tmp_path: Path) -> None:
    expanded = expanded_dataset()
    moved = [atoms.copy() for atoms in expanded.structures]
    moved[3].set_positions(moved[3].get_positions() + 0.01)
    hashes: list[str | None] = []
    content_hashes: list[str] = []
    for name, structures in (("a", expanded.structures), ("b", moved)):
        written = write_dataset(
            expanded.spec,
            expanded.table,
            structures,
            expanded.provenance,
            tmp_path / name,
            layout=SMALL_ZARR_LAYOUT,
        )
        assert written.outputs is not None
        hashes.append(written.outputs.structures_sha256)
        content_hashes.append(written.outputs.table_parquet.content_sha256)
    assert hashes[0] != hashes[1]
    # the table did not change, so neither did its content hash
    assert content_hashes[0] == content_hashes[1]


def test_zarr_problems_compares_the_zarr_to_the_table_row_for_row(
    tmp_path: Path,
) -> None:
    expanded = expanded_dataset()
    directory = tmp_path / DATASET_ID
    write_dataset(
        expanded.spec,
        expanded.table,
        expanded.structures,
        expanded.provenance,
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    spec, table = read_bundle(directory).spec, read_bundle(directory).table

    shorter = table.iloc[:-1]
    assert zarr_problems(directory, spec, shorter) == [
        "8: the zarr holds 6 structures for 5 rows"
    ]
    changed_ids = table.copy()
    changed_ids["molecule_id"] = changed_ids["molecule_id"][::-1].to_numpy()
    assert zarr_problems(directory, spec, changed_ids) == [
        "8: ids/molecule_id does not match the table"
    ]
    changed_split = table.copy()
    changed_split.loc[0, "split"] = "test"
    assert zarr_problems(directory, spec, changed_split) == [
        "8: tasks/split does not match the table's default split"
    ]
    changed_label = table.copy()
    changed_label.loc[0, "logp"] = 99.0
    assert zarr_problems(directory, spec, changed_label) == [
        "8: the zarr's labels do not match the table"
    ]
    changed_mask = table.copy()
    changed_mask.loc[0, "activity"] = np.nan
    assert zarr_problems(directory, spec, changed_mask) == [
        "8: the zarr's labels do not match the table"
    ]


def test_a_labelled_corpus_without_evaluation_is_written_unassigned(
    tmp_path: Path,
) -> None:
    expanded = expanded_dataset(evaluation=False)
    directory = tmp_path / "corpus"
    write_dataset(
        expanded.spec,
        expanded.table,
        expanded.structures,
        expanded.provenance,
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
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
    spec = make_dataset_spec(
        dataset_id="source_like",
        labels=[LabelColumn(name="gap", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
        extra_columns=["csd_code"],
    ).model_copy(update={"smiles": False, "geometry_origin": "source"})
    table = make_dataset_table(
        make_dataset_spec(
            dataset_id="source_like",
            labels=[LabelColumn(name="gap", task_type=TaskType.regression)],
            metrics=[EvalMetric.mae],
        ),
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


def test_a_charged_molecule_keeps_its_total_charge(tmp_path: Path) -> None:
    smiles_values = ["OCC[NH3+]", "CC(=O)[O-]", "CCO"]
    spec = make_dataset_spec(
        dataset_id="charged",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
    )
    table = make_dataset_table(
        spec,
        smiles_values,
        np.zeros(3),
        {"split": ["train", "valid", "test"]},
    )
    assert table["total_charge"].tolist() == [1.0, -1.0, 0.0]
    directory = tmp_path / "charged"
    write_dataset(
        spec,
        table,
        [embed_frame(smiles) for smiles in smiles_values],
        make_provenance("charged"),
        directory,
        layout=SMALL_ZARR_LAYOUT,
    )
    dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert np.asarray(dataset.total_charge[:]).tolist() == [1.0, -1.0, 0.0]
        assert np.asarray(dataset.multiplicity[:]).tolist() == [1.0, 1.0, 1.0]
    finally:
        dataset.close()


# -------------------------------------------------------------- build_datasets


def test_build_datasets_builds_and_verifies_every_bundle(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    write_smiles_bundle(tmp_path / "bundles", dataset_id="second")
    config = build_config(tmp_path)

    report = build_datasets(config)

    assert report.n_failed == 0
    assert [(status.kind, status.ok) for status in report.statuses] == [
        ("build_dataset", True),
        ("build_dataset", True),
        ("verify_dataset", True),
        ("verify_dataset", True),
    ]
    assert [status.name for status in report.statuses] == [
        "0_build_dataset_second",
        "1_build_dataset_synthetic6",
        "2_verify_dataset_second",
        "3_verify_dataset_synthetic6",
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
    assert (zarr_root / "verify" / f"{DATASET_ID}.yaml").is_file()

    directory = zarr_root / DATASET_ID
    dataset = read_bundle(directory)
    assert dataset.spec.geometry_origin == "etkdg_mmff"
    assert len(dataset.table) == 6
    assert (directory / TIMINGS_FILENAME).is_file()
    timings = (directory / TIMINGS_FILENAME).read_text().splitlines()
    assert len(timings) == 6

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


def test_a_rerun_skips_a_dataset_built_from_the_same_bundle(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path)
    assert build_datasets(config).n_failed == 0
    table_path = tmp_path / "zarrs" / DATASET_ID / TABLE_FILENAME
    before = table_path.stat().st_mtime_ns

    report = build_datasets(config)

    assert report.n_failed == 0
    assert table_path.stat().st_mtime_ns == before
    summary = read_summary(config, DATASET_ID)
    assert summary.skipped
    assert summary.structures_out == 0


def test_changed_conformer_settings_rebuild(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    assert build_datasets(build_config(tmp_path)).n_failed == 0

    config = build_config(tmp_path, conformers=ConformerEmbeddingConfig(n_conformers=2))
    assert build_datasets(config).n_failed == 0

    summary = read_summary(config, DATASET_ID)
    assert not summary.skipped
    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert len(dataset.table) == 12
    assert dataset.provenance.conformers is not None
    assert dataset.provenance.conformers.settings.n_conformers == 2


def test_changed_geometry_limits_rebuild(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    assert build_datasets(build_config(tmp_path)).n_failed == 0

    limits = DEFAULT_GEOMETRY_LIMITS.model_copy(
        update={"min_interatomic_distance": 0.4}
    )
    config = build_config(tmp_path, geometry_limits=limits)
    assert build_datasets(config).n_failed == 0

    assert not read_summary(config, DATASET_ID).skipped
    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert dataset.provenance.geometry_limits == limits


def test_a_changed_bundle_rebuilds(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path)
    assert build_datasets(config).n_failed == 0
    write_smiles_bundle(
        tmp_path / "bundles", dataset_id=DATASET_ID, require_enantiomer_pairs=True
    )

    assert build_datasets(config).n_failed == 0

    assert not read_summary(config, DATASET_ID).skipped
    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert dataset.spec.evaluation is not None
    assert dataset.spec.evaluation.require_enantiomer_pairs


def test_overwrite_rebuilds_an_up_to_date_dataset(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    assert build_datasets(build_config(tmp_path)).n_failed == 0

    config = build_config(tmp_path, overwrite=True)
    assert build_datasets(config).n_failed == 0

    summary = read_summary(config, DATASET_ID)
    assert not summary.skipped
    assert summary.structures_out == 6


def test_dataset_ids_select_the_bundles_to_build(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    write_smiles_bundle(tmp_path / "bundles", dataset_id="ignored")
    config = build_config(tmp_path, dataset_ids=[DATASET_ID])
    assert config.resolve_dataset_ids() == [DATASET_ID]

    assert build_datasets(config).n_failed == 0

    assert (tmp_path / "zarrs" / DATASET_ID).is_dir()
    assert not (tmp_path / "zarrs" / "ignored").exists()


def test_nested_bundles_keep_their_relative_path(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles" / "tdc", dataset_id="AMES")
    config = build_config(tmp_path, verify=False)
    assert config.resolve_dataset_ids() == ["tdc/AMES"]

    report = build_datasets(config)

    assert report.n_failed == 0
    assert [status.name for status in report.statuses] == ["0_build_dataset_tdc__AMES"]
    assert read_bundle(tmp_path / "zarrs" / "tdc" / "AMES").spec.dataset_id == "AMES"


def test_one_failing_bundle_does_not_stop_the_others(tmp_path: Path) -> None:
    write_smiles_bundle(
        tmp_path / "bundles", dataset_id="a_without_smiles", smiles=False
    )
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path)

    report = build_datasets(config)

    assert report.n_failed == 1
    failed, built, verified = report.statuses
    assert not failed.ok and "smiles: false" in (failed.error or "")
    assert built.ok and verified.ok
    assert verified.name.endswith(DATASET_ID)
    assert not (tmp_path / "zarrs" / "a_without_smiles").exists()
    status = yaml.safe_load((tmp_path / "zarrs" / "status.yaml").read_text())
    assert status["n_failed"] == 1


def test_keep_going_false_raises_but_still_writes_the_status(tmp_path: Path) -> None:
    write_smiles_bundle(
        tmp_path / "bundles", dataset_id="a_without_smiles", smiles=False
    )
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path, keep_going=False)

    with pytest.raises(ValueError, match="smiles: false"):
        build_datasets(config)

    assert (tmp_path / "zarrs" / "status.yaml").is_file()
    assert not (tmp_path / "zarrs" / DATASET_ID).exists()


def test_keep_going_false_records_the_failure_in_the_status(tmp_path: Path) -> None:
    write_smiles_bundle(
        tmp_path / "bundles", dataset_id="a_without_smiles", smiles=False
    )
    config = build_config(tmp_path, keep_going=False)

    with pytest.raises(ValueError, match="smiles: false"):
        build_datasets(config)

    status = yaml.safe_load((tmp_path / "zarrs" / "status.yaml").read_text())
    assert status["n_failed"] == 1


def test_verify_covers_datasets_a_preparer_wrote_directly(tmp_path: Path) -> None:
    write_small_dataset(
        tmp_path / "zarrs",
        dataset_id="direct",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 1.0, 10),
    )
    (tmp_path / "bundles").mkdir()

    report = build_datasets(build_config(tmp_path))

    assert report.n_failed == 0
    assert [status.name for status in report.statuses] == ["0_verify_dataset_direct"]


def test_a_failed_verification_is_a_failed_task(tmp_path: Path) -> None:
    directory = write_small_dataset(
        tmp_path / "zarrs",
        dataset_id="direct",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 1.0, 10),
    )
    tamper_provenance_structures_hash(directory)
    (tmp_path / "bundles").mkdir()

    report = build_datasets(build_config(tmp_path))

    assert report.n_failed == 1
    (status,) = report.statuses
    assert "structures_sha256" in (status.error or "")
    assert (tmp_path / "zarrs" / "verify" / "direct.yaml").is_file()


def failing_embedding_for(smiles: str):
    """An ``embed_one_smiles`` that reports ``embed_failed`` for one molecule."""
    target = canonical(smiles)
    real_embed_one_smiles = conformers_module.embed_one_smiles

    def embed_one_smiles(
        isomeric_smiles: str, config: ConformerEmbeddingConfig
    ) -> EmbedResult:
        if canonical(isomeric_smiles) != target:
            return real_embed_one_smiles(isomeric_smiles, config)
        return EmbedResult(
            nonisomeric_smiles=None,
            positions=None,
            atomic_numbers=None,
            timing=ConformerTimingRecord(
                isomeric_smiles=isomeric_smiles,
                n_atoms=-1,
                n_confs_requested=config.n_conformers,
                n_confs_emitted=0,
                t_embed_s=0.0,
                t_mmff_s=0.0,
                status="embed_failed",
                error_msg="injected failure",
            ),
        )

    return embed_one_smiles


def test_an_embedding_failure_nulls_the_surviving_partner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        conformers_module, "embed_one_smiles", failing_embedding_for(D_ALANINE)
    )
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)

    assert build_datasets(build_config(tmp_path)).n_failed == 0

    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert len(dataset.table) == 5
    assert canonical(D_ALANINE) not in set(dataset.table["isomeric_smiles"])
    orphan = dataset.table[dataset.table["isomeric_smiles"] == canonical(L_ALANINE)]
    assert orphan["enantiomer_of"].isna().all()
    dropped = dataset.provenance.counts.dropped
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 1
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 1
    assert dropped[STEREO_MISMATCH_AFTER_EMBEDDING] == 0


def test_an_embedding_failure_drops_the_orphan_when_pairs_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        conformers_module, "embed_one_smiles", failing_embedding_for(D_ALANINE)
    )
    write_smiles_bundle(
        tmp_path / "bundles", dataset_id=DATASET_ID, require_enantiomer_pairs=True
    )

    assert build_datasets(build_config(tmp_path)).n_failed == 0

    dataset = read_bundle(tmp_path / "zarrs" / DATASET_ID)
    assert len(dataset.table) == 4
    assert dataset.table["enantiomer_of"].notna().all()
    dropped = dataset.provenance.counts.dropped
    assert dropped[CONFORMER_EMBEDDING_FAILED] == 1
    assert dropped[ENANTIOMER_PARTNER_FAILED] == 1


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


def test_frames_outside_the_geometry_limits_are_dropped_and_counted(
    tmp_path: Path,
) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id=DATASET_ID)
    config = build_config(tmp_path)
    config = config.model_copy(
        update={
            "geometry_limits": config.geometry_limits.model_copy(
                update={"max_atoms": 13}
            )
        }
    )

    assert build_datasets(config).n_failed == 0

    directory = tmp_path / "zarrs" / DATASET_ID
    dataset = read_bundle(directory)
    # alanine (13 atoms), phenol (13) and ethanol (9) fit; tartaric acid (16)
    # and 2-butanol (15) do not
    assert len(dataset.table) == 4
    assert dataset.provenance.counts.dropped["max_atoms"] == 2
    assert dataset.provenance.geometry_limits.max_atoms == 13
    molecule_dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        assert all(len(atoms) <= 13 for atoms in molecule_dataset.get_all_molecules())
    finally:
        molecule_dataset.close()


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


def test_a_charged_molecule_is_built_with_its_total_charge(tmp_path: Path) -> None:
    bundle = make_bundle()
    table = bundle.table.copy()
    table["isomeric_smiles"] = table["isomeric_smiles"].astype(object)
    table["nonisomeric_smiles"] = table["nonisomeric_smiles"].astype(object)
    ethanol_row = table.index[table["isomeric_smiles"] == "CCO"][0]
    table.loc[ethanol_row, "isomeric_smiles"] = "NCC[NH3+]"
    table.loc[ethanol_row, "nonisomeric_smiles"] = "NCC[NH3+]"
    bundle.table = table
    from remedi.data_handling.bundle import write_bundle

    write_bundle(bundle, tmp_path / "bundles" / DATASET_ID)

    assert build_datasets(build_config(tmp_path)).n_failed == 0

    directory = tmp_path / "zarrs" / DATASET_ID
    dataset = read_bundle(directory)
    charged = dataset.table["isomeric_smiles"] == "NCC[NH3+]"
    assert dataset.table.loc[charged, "total_charge"].tolist() == [1.0]
    assert (dataset.table.loc[~charged, "total_charge"] == 0.0).all()
    molecule_dataset = MoleculeDataset.open_existing_dataset_from_dir(directory)
    try:
        total_charge = np.asarray(molecule_dataset.total_charge[:])
        assert total_charge.tolist() == dataset.table["total_charge"].tolist()
    finally:
        molecule_dataset.close()


# -------------------------------------------------------------- verify_dataset


def built_dataset(tmp_path: Path) -> Path:
    return write_small_dataset(
        tmp_path,
        dataset_id="ten",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 1.0, 10),
        splits={"split": list(TEN_ROW_SPLIT)},
    )


def tamper_provenance_structures_hash(directory: Path) -> None:
    provenance_path = directory / PROVENANCE_FILENAME
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["structures_sha256"] = "0" * 64
    provenance_path.write_text(yaml.safe_dump(document, sort_keys=False))


def test_verify_passes_on_a_fresh_dataset(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    report = verify_dataset(directory)
    assert report.problems == []
    assert report.dataset_id == "ten"
    assert report.rows == 10
    assert report.structures_sha256 == structures_identity(directory)


def test_verify_catches_tampered_zarr_positions(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    positions = zarr.open_array(directory / "positions", mode="r+")
    shifted = np.asarray(positions[:])
    shifted[1] = shifted[0]  # two atoms of the first molecule on top of each other
    positions[:] = shifted

    problems = verify_dataset(directory).problems

    assert any("structures_sha256" in problem for problem in problems), problems
    assert any(problem.startswith("10:") for problem in problems), problems


def test_verify_catches_a_tampered_zarr_id_array(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    molecule_ids = zarr.open_array(directory / "ids" / "molecule_id", mode="r+")
    molecule_ids[:] = np.asarray(molecule_ids[:])[::-1]

    problems = verify_dataset(directory).problems

    assert problems == ["8: ids/molecule_id does not match the table"]


def test_verify_catches_a_tampered_table(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    table = pd.read_parquet(directory / TABLE_FILENAME)
    table.loc[0, "y"] = 42.0
    table.to_parquet(directory / TABLE_FILENAME, index=False)

    report = verify_dataset(directory)

    assert len(report.problems) == 1
    assert "content_sha256" in report.problems[0]


def test_verify_catches_a_tampered_provenance_hash(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    tamper_provenance_structures_hash(directory)

    problems = verify_dataset(directory).problems

    assert len(problems) == 1 and "structures_sha256" in problems[0], problems


def test_verify_reports_a_missing_file_instead_of_raising(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    (directory / TABLE_FILENAME).unlink()
    report = verify_dataset(directory)
    assert report.dataset_id is None
    assert len(report.problems) == 1 and TABLE_FILENAME in report.problems[0]


def test_the_verify_task_yields_its_report_then_raises(tmp_path: Path) -> None:
    directory = built_dataset(tmp_path)
    tamper_provenance_structures_hash(directory)
    task = VerifyDatasetTask(dataset_path=directory)
    results = task.run(None)

    first = next(results)
    assert isinstance(first, PydanticResult)
    assert isinstance(first.obj, VerifyReport)
    with pytest.raises(ValueError, match="failed verification"):
        next(results)
