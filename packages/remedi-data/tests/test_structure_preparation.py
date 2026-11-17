"""The structure preparers (``remedi_data.preparation.structures``) on small synthetic raw files.

Each source is exercised through its real reader on files written here in the
source's raw format (tmQM xyz.gz + csv, a PCQM-style SDF + csv.gz, a GEOM-style
rdkit_folder of pickles), and the pipeline end to end: shards, restart, ids,
drop counts and the written dataset passing ``verify_dataset``.
"""

from __future__ import annotations

import gzip
import itertools
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pydantic
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from remedi_data.chemistry.elements import ElementSet
from remedi_data.chemistry.geometry import GeometryLimits
from remedi_data.chemistry.smiles_filter import (
    SmilesFilterConfig,
    molecule_passes_filter,
)
from remedi_data.dataset.structures import Structures
from remedi_data.format import (
    TABLE_FILENAME,
    assign_identity,
    identity_from_canonical,
    mirror_isomeric_smiles,
    read_bundle,
    read_provenance,
)
from remedi_data.format.verify import verify_dataset
from remedi_data.preparation.structures import (
    DroppedRecord,
    GeomDrugsSource,
    Pcqm4mv2Source,
    PreparedFrame,
    ShardFailures,
    StructurePreparationConfig,
    StructurePreparationSettings,
    StructureRecord,
    prepare_record,
    prepare_structure_datasets,
    prepare_structures,
)
from remedi_data.preparation.structures.pipeline import run_shards
from remedi_data.preparation.structures.records import (
    CONNECTIVITY_DIFFERS_FROM_SOURCE,
    SMILES_FILTER,
    UNREADABLE,
    round_trip_canonical_smiles,
)
from remedi_data.preparation.structures.sources.pcqm4mv2 import sdf_record_offsets
from remedi_data.preparation.structures.sources.tmqm import (
    iterate_xyz_entries,
    parse_comment,
)
from remedi_data.yaml_io import read_model

SETTINGS = StructurePreparationSettings(shard_size=3, workers=1)
FILTER = SmilesFilterConfig(
    max_atoms=100, element_set=ElementSet.mace_off, neutralize=False, dedupe=False
)
LIMITS = GeometryLimits(elements=ElementSet.mace_off, min_interatomic_distance=0.5)


def embedded_molecule(smiles: str, seed: int = 7, shuffle: bool = True) -> Chem.Mol:
    """An MMFF-relaxed molecule with explicit hydrogens, atoms in a shuffled order.

    The shuffle stands in for a source whose atom order is not RDKit's
    ``AddHs`` order of the SMILES (hydrogens first here).
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = seed
    assert rdDistGeom.EmbedMolecule(molecule, parameters) == 0
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule)
    if not shuffle:
        return molecule
    order = list(range(molecule.GetNumAtoms()))[::-1]
    return Chem.RenumberAtoms(molecule, order)


def prepare(record: StructureRecord, smiles_filter=FILTER, limits=LIMITS):
    return prepare_record(
        record,
        smiles_filter=smiles_filter,
        limits=limits,
        allowed_atomic_numbers=limits.allowed_atomic_numbers(),
    )


# ------------------------------------------------------------ Structures


def test_structures_slice_take_concatenate_and_npz(tmp_path: Path) -> None:
    frames = [
        (np.array([6, 1]), np.arange(6.0).reshape(2, 3)),
        (np.array([8]), np.ones((1, 3))),
        (np.array([7, 1, 1]), np.full((3, 3), 2.0)),
    ]
    packed = Structures.from_frames(frames)
    assert len(packed) == 3 and packed.ptr.tolist() == [0, 2, 3, 6]
    assert packed.positions.dtype == np.float32
    assert packed.atomic_numbers.dtype == np.uint8
    taken = packed.take([2, 0])
    assert taken.ptr.tolist() == [0, 3, 5]
    assert taken.atomic_numbers.tolist() == [7, 1, 1, 6, 1]
    sliced = packed.slice(1, 3)
    assert sliced.ptr.tolist() == [0, 1, 4]
    joined = Structures.concatenate([packed.slice(0, 1), sliced])
    assert np.array_equal(joined.positions, packed.positions)
    assert joined.ptr.tolist() == packed.ptr.tolist()
    packed.save_npz(tmp_path / "frames.npz")
    loaded = Structures.load_npz(tmp_path / "frames.npz")
    assert np.array_equal(loaded.positions, packed.positions)
    assert packed.to_atoms()[2].get_chemical_symbols() == ["N", "H", "H"]
    assert len(Structures.concatenate([])) == 0
    with pytest.raises(ValueError, match="ptr"):
        Structures.from_arrays(np.zeros((2, 3)), np.zeros(2), np.array([1, 2]))


# ------------------------------------------------------------------- identity


def test_identity_from_canonical_matches_assign_identity() -> None:
    smiles = [
        "C[C@H](N)C(=O)O",
        "C[C@@H](N)C(=O)O",
        "CC(N)C(=O)O",
        "C[C@H](N)C(=O)O",
        "C[C@H](O)[C@@H](C)O",  # meso: its own mirror image
        "CCO",
    ]
    expected = assign_identity(smiles).to_frame()
    canonical = expected["isomeric_smiles"].tolist()
    actual = identity_from_canonical(
        canonical,
        expected["nonisomeric_smiles"].tolist(),
        [mirror_isomeric_smiles(value) for value in canonical],
    ).to_frame()
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["enantiomer_of"].tolist()[:2] == [1, 0]
    assert pd.isna(actual["enantiomer_of"].iloc[4])


#: Found in PCQM4Mv2: RDKit keeps tags on the symmetric cage atoms of this
#: achiral molecule; writing the first spelling and parsing it again gives the
#: second, which is a fixed point and its own mirror image.
CAGE_SPELLING = "O=CCC[C@H]1[C@@H]2CC[C@H]1CC2"
CAGE_FIXED_POINT = "O=CCC[C@H]1[C@H]2CC[C@@H]1CC2"


def test_a_one_sided_mirror_relation_is_not_a_pair() -> None:
    assert mirror_isomeric_smiles(CAGE_SPELLING) == CAGE_FIXED_POINT
    assert mirror_isomeric_smiles(CAGE_FIXED_POINT) == CAGE_FIXED_POINT
    smiles = [CAGE_SPELLING, CAGE_FIXED_POINT, "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]
    canonical_alanines = [Chem.CanonSmiles(value) for value in smiles[2:]]
    identity = identity_from_canonical(
        [*smiles[:2], *canonical_alanines],
        ["O=CCCC1C2CCC1CC2"] * 2 + ["CC(N)C(=O)O"] * 2,
        [mirror_isomeric_smiles(value) for value in [*smiles[:2], *canonical_alanines]],
    )
    assert identity.enantiomer_of.tolist() == [pd.NA, pd.NA, 3, 2]


def test_the_stored_smiles_is_a_round_trip_fixed_point() -> None:
    assert round_trip_canonical_smiles(CAGE_SPELLING) is not None
    assert round_trip_canonical_smiles(CAGE_SPELLING)[0] == CAGE_FIXED_POINT
    assert round_trip_canonical_smiles("not a smiles") is None


# -------------------------------------------------------------- prepare_record


def test_a_source_order_frame_is_reordered_to_the_smiles() -> None:
    molecule = embedded_molecule("C[C@H](N)C(=O)O")
    prepared = prepare(StructureRecord(source_key="a", molecule=molecule))
    assert isinstance(prepared, PreparedFrame)
    assert prepared.isomeric_smiles == Chem.CanonSmiles("C[C@H](N)C(=O)O")
    assert prepared.nonisomeric_smiles == "CC(N)C(=O)O"
    assert prepared.mirror_smiles == Chem.CanonSmiles("C[C@@H](N)C(=O)O")
    template = Chem.AddHs(Chem.MolFromSmiles(prepared.isomeric_smiles))
    assert prepared.atomic_numbers.tolist() == [
        atom.GetAtomicNum() for atom in template.GetAtoms()
    ]
    # The same atoms, reordered: the set of interatomic distances is unchanged.
    source_positions = molecule.GetConformer().GetPositions()
    distances = np.sort(
        np.linalg.norm(source_positions[:, None] - source_positions[None], axis=-1),
        axis=None,
    )
    reordered = prepared.positions.astype(float)
    reordered_distances = np.sort(
        np.linalg.norm(reordered[:, None] - reordered[None], axis=-1), axis=None
    )
    assert np.allclose(distances, reordered_distances, atol=1e-5)
    assert prepared.total_charge == 0.0 and prepared.multiplicity == 1.0


def test_the_enantiomer_frame_gets_the_mirror_smiles() -> None:
    molecule = embedded_molecule("C[C@@H](N)C(=O)O")
    prepared = prepare(StructureRecord(source_key="b", molecule=molecule))
    assert isinstance(prepared, PreparedFrame)
    assert prepared.isomeric_smiles == Chem.CanonSmiles("C[C@@H](N)C(=O)O")


def test_a_charged_molecule_keeps_its_charge() -> None:
    prepared = prepare(
        StructureRecord(source_key="c", molecule=embedded_molecule("CC(=O)[O-]"))
    )
    assert isinstance(prepared, PreparedFrame)
    assert prepared.total_charge == -1.0


def test_a_zwitterion_keeps_its_charges() -> None:
    record = StructureRecord(
        source_key="z", molecule=embedded_molecule("[NH3+]CC(=O)[O-]")
    )
    prepared = prepare(record)
    assert isinstance(prepared, PreparedFrame)
    assert prepared.isomeric_smiles == "[NH3+]CC(=O)[O-]"
    assert prepared.total_charge == 0.0


@pytest.mark.parametrize(
    "setting",
    [{"neutralize": True}, {"dedupe": True}],
    ids=["neutralize", "dedupe"],
)
@pytest.mark.parametrize("source", [Pcqm4mv2Source, GeomDrugsSource])
def test_a_structure_source_refuses_neutralize_and_dedupe(
    tmp_path: Path, source, setting: dict[str, bool]
) -> None:
    """Neutralising changes the hydrogen count, so the SMILES no longer describes
    the frame; deduplicating would need state across records the source cannot keep."""
    paths = (
        {"sdf": tmp_path / "a.sdf", "labels_csv": tmp_path / "a.csv"}
        if source is Pcqm4mv2Source
        else {"directory": tmp_path}
    )
    with pytest.raises(pydantic.ValidationError, match=next(iter(setting))):
        source(**paths, smiles_filter=FILTER.model_copy(update=setting))
    assert source(**paths, smiles_filter=FILTER).smiles_filter == FILTER


def test_a_salt_and_a_foreign_element_are_dropped_by_the_filter() -> None:
    salt = StructureRecord(
        source_key="s", molecule=embedded_molecule("CC(=O)[O-].[Na+]")
    )
    silane = StructureRecord(
        source_key="si", molecule=embedded_molecule("C[Si](C)(C)C")
    )
    assert prepare(salt) == SMILES_FILTER
    assert prepare(silane) == SMILES_FILTER


def test_array_records_pass_the_geometry_gate_only() -> None:
    record = StructureRecord(
        source_key="x",
        atomic_numbers=np.array([26, 6, 8]),
        positions=np.array([[0.0, 0, 0], [1.8, 0, 0], [2.9, 0, 0]]),
        total_charge=1.0,
        multiplicity=2.0,
    )
    prepared = prepare(record, smiles_filter=None, limits=GeometryLimits())
    assert isinstance(prepared, PreparedFrame) and prepared.isomeric_smiles is None
    assert prepare(record, smiles_filter=None, limits=LIMITS) == "element_gate"
    incomplete = StructureRecord(source_key="y", atomic_numbers=np.array([6]))
    assert prepare(incomplete, smiles_filter=None) == UNREADABLE
    assert prepare(StructureRecord(source_key="m"), smiles_filter=FILTER) == UNREADABLE


def test_a_trusted_graph_replaces_perceived_bond_orders_and_charges() -> None:
    # GEOM's rd_mol sometimes carries a charge-separated form of the molecule.
    record = StructureRecord(
        source_key="dmso",
        molecule=embedded_molecule("C[S+](C)[O-]"),
        graph_smiles="CS(C)=O",
    )
    prepared = prepare(record)
    assert isinstance(prepared, PreparedFrame)
    assert prepared.isomeric_smiles == "CS(C)=O"
    assert prepared.total_charge == 0.0


def test_a_frame_whose_hydrogen_moved_is_dropped() -> None:
    # 2-hydroxypyridine geometry, but the source says 2-pyridone.
    record = StructureRecord(
        source_key="tautomer",
        molecule=embedded_molecule("Oc1ccccn1"),
        graph_smiles="O=c1cccc[nH]1",
    )
    assert prepare(record) == CONNECTIVITY_DIFFERS_FROM_SOURCE
    unparsable = StructureRecord(
        source_key="bad", molecule=embedded_molecule("CCO"), graph_smiles="C1CC"
    )
    assert prepare(unparsable) == SMILES_FILTER


def test_the_filter_runs_on_the_trusted_graph_first() -> None:
    record = StructureRecord(
        source_key="salt",
        molecule=embedded_molecule("CC(=O)O"),
        graph_smiles="C[Si](C)(C)C",
    )
    assert prepare(record) == SMILES_FILTER


# ----------------------------------------------------------------- PCQM-style


PCQM_SMILES = [
    "C[C@H](N)C(=O)O",
    "C[C@@H](N)C(=O)O",
    "CCO",
    "C[Si](C)(C)C",
    "c1ccccc1O",
    "CCO",
    "CC(=O)[O-]",
]


@pytest.fixture
def pcqm_files(tmp_path: Path) -> tuple[Path, Path]:
    sdf = tmp_path / "raw" / "train.sdf"
    sdf.parent.mkdir()
    writer = Chem.SDWriter(str(sdf))
    for index, smiles in enumerate(PCQM_SMILES):
        writer.write(embedded_molecule(smiles, seed=index))
    writer.close()
    labels = tmp_path / "raw" / "data.csv.gz"
    pd.DataFrame(
        {
            "idx": range(len(PCQM_SMILES) + 2),
            "smiles": [*PCQM_SMILES, "C", "N"],
            "homolumogap": [float(index) for index in range(len(PCQM_SMILES) + 2)],
        }
    ).to_csv(labels, index=False)
    return sdf, labels


def test_sdf_record_offsets_split_the_file_at_record_ends(pcqm_files) -> None:
    sdf, _ = pcqm_files
    offsets = sdf_record_offsets(sdf)
    assert len(offsets) == len(PCQM_SMILES) + 1
    data = sdf.read_bytes()
    assert offsets[-1] == len(data)
    for start, stop in itertools.pairwise(offsets):
        assert data[start:stop].rstrip().endswith(b"$$$$")


def test_pcqm_source_end_to_end(tmp_path: Path, pcqm_files) -> None:
    sdf, labels = pcqm_files
    source = Pcqm4mv2Source(sdf=sdf, labels_csv=labels, smiles_filter=FILTER)
    output = tmp_path / "datasets" / "pcqm4mv2"
    summary = prepare_structures(source, SETTINGS, output, tmp_path / "work")
    assert summary.records_read == len(PCQM_SMILES)
    assert summary.dropped == {SMILES_FILTER: 1}
    assert summary.shards == 3
    bundle = read_bundle(output)
    table = bundle.table
    assert table["source_key"].tolist() == ["0", "1", "2", "4", "5", "6"]
    assert table["homo_lumo_gap"].tolist() == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]
    assert table["total_charge"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    # The two alanines are enantiomers; the two ethanols share every id.
    assert table["enantiomer_of"].tolist()[:2] == [1, 0]
    assert table["molecule_id"].iloc[0] == table["molecule_id"].iloc[1]
    assert table["stereoisomer_id"].iloc[2] == table["stereoisomer_id"].iloc[4]
    provenance = read_provenance(output)
    assert provenance.counts.source_molecules == len(PCQM_SMILES)
    assert provenance.counts.dropped == {SMILES_FILTER: 1}
    assert set(provenance.source.files) == {"train.sdf", "data.csv.gz"}
    assert bundle.spec.smiles and bundle.spec.geometry_origin == "source"
    assert verify_dataset(output).problems == []
    # The shards were streamed into the dataset: one table row group per shard
    # (3, 2 after the dropped record, 1 rows), records in shard order.
    row_groups = pq.ParquetFile(output / TABLE_FILENAME).metadata
    assert [
        row_groups.row_group(index).num_rows
        for index in range(row_groups.num_row_groups)
    ] == [3, 2, 1]
    # A second call with the same config does nothing.
    assert prepare_structures(source, SETTINGS, output, tmp_path / "work").skipped


def test_a_raw_file_replaced_at_the_same_path_rebuilds_the_dataset(
    tmp_path: Path, pcqm_files
) -> None:
    sdf, labels = pcqm_files
    source = Pcqm4mv2Source(sdf=sdf, labels_csv=labels, smiles_filter=FILTER)
    output = tmp_path / "datasets" / "pcqm4mv2"
    prepare_structures(source, SETTINGS, output, tmp_path / "work")

    relabelled = pd.read_csv(labels)
    relabelled["homolumogap"] += 10.0
    relabelled.to_csv(labels, index=False)

    # The shards of the old file are refused rather than mixed with the new one...
    with pytest.raises(ValueError, match="raw files"):
        prepare_structures(source, SETTINGS, output, tmp_path / "work")
    # ...and a fresh work directory rebuilds the dataset from the new file.
    summary = prepare_structures(source, SETTINGS, output, tmp_path / "fresh_work")
    assert not summary.skipped
    assert read_bundle(output).table["homo_lumo_gap"].tolist()[0] == 10.0
    assert prepare_structures(source, SETTINGS, output, tmp_path / "fresh_work").skipped


# ------------------------------------------------------------- restart, plan


class _FailingSource(Pcqm4mv2Source):
    """Fails on records 3:6 while the class-level switch is on (not part of the config)."""

    failing: ClassVar[bool] = True

    def read_records(
        self, start: int, stop: int, work_directory: Path
    ) -> Iterator[StructureRecord | DroppedRecord]:
        if type(self).failing and start == 3:
            raise RuntimeError("disk hiccup")
        yield from super().read_records(start, stop, work_directory)


def test_a_failed_shard_is_retried_and_finished_shards_are_reused(
    tmp_path: Path, pcqm_files
) -> None:
    sdf, labels = pcqm_files
    work = tmp_path / "work"
    failing = _FailingSource(sdf=sdf, labels_csv=labels, smiles_filter=FILTER)
    with pytest.raises(ShardFailures, match="shard 1"):
        run_shards(failing, SETTINGS, work)
    assert sorted(path.name for path in (work / "shards").glob("*.yaml")) == [
        "00000.yaml",
        "00002.yaml",
    ]
    # Another config is refused: its shards would mix two filters...
    other = failing.model_copy(update={"limit": 5})
    with pytest.raises(ValueError, match="another source config"):
        run_shards(other, SETTINGS, work)
    # ...while the same config reruns only the failed shard.
    _FailingSource.failing = False
    try:
        summaries, reused = run_shards(failing, SETTINGS, work)
    finally:
        _FailingSource.failing = True
    assert reused == 2 and [summary.index for summary in summaries] == [0, 1, 2]


# ------------------------------------------------------------------- tmQM-style


TMQM_ENTRIES = [
    ("AAAAAA", 0, 0, [("Fe", 0.0), ("C", 1.9), ("O", 3.05), ("H", -1.6)]),
    ("BBBBBB", 1, 0, [("Cu", 0.0), ("N", 2.0), ("H", 3.0)]),
    ("CCCCCC", 0, 0, [("Ni", 0.0), ("C", 0.2), ("H", 1.3)]),  # overlapping atoms
    ("DDDDDD", -1, 0, [("Zn", 0.0), ("Cl", 2.2), ("H", -1.6)]),
    ("EEEEEE", 0, 0, [("Zn", 0.0), ("Cl", 2.2), ("Cl", -2.2)]),  # no hydrogen
]


def write_tmqm_directory(directory: Path) -> None:
    directory.mkdir(parents=True)
    blocks = []
    for code, charge, spin, atoms in TMQM_ENTRIES:
        lines = [str(len(atoms))]
        lines.append(
            f"CSD_code = {code} | q = {charge} | S = {spin} | Stoichiometry = X | "
            "MND = 4 | 2020-2024 CSD"
        )
        lines += [f"{symbol}  {x:.5f}  0.00000  0.00000" for symbol, x in atoms]
        blocks.append("\n".join(lines) + "\n")
    # Entries separated by blank lines, as in the release files.
    for index, name in enumerate(
        ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz")
    ):
        chunk = blocks[:2] if index == 0 else blocks[2:3] if index == 1 else blocks[3:]
        with gzip.open(directory / name, "wt") as handle:
            handle.write("\n".join(chunk) + "\n")
    rows = [
        ";".join(
            [code]
            + [str(float(index * 10 + column)) for column in range(8)]
            + ["2020-2024", "" if code == "BBBBBB" else "[Fe]C=O"]
        )
        for index, (code, *_rest) in enumerate(TMQM_ENTRIES)
        if code not in ("DDDDDD", "EEEEEE")  # complexes without labels
    ]
    header = (
        "CSD_code;Electronic_E;Dispersion_E;Dipole_M;Metal_q;HL_Gap;HOMO_Energy;"
        "LUMO_Energy;Polarizability;CSD_years;SMILES"
    )
    (directory / "tmQM_y.csv").write_text("\n".join([header, *rows]) + "\n")


def test_xyz_parser_skips_blank_lines_and_parses_the_comment() -> None:
    lines = iter(["\n", "1\n", "a = 1 | b = x | free text\n", "H 0 0 0\n", "\n"])
    entries = list(iterate_xyz_entries(lines))
    assert entries == [("a = 1 | b = x | free text", ["H 0 0 0\n"])]
    assert parse_comment(entries[0][0]) == {"a": "1", "b": "x"}


def test_tmqm_source_through_the_config_and_a_process_pool(tmp_path: Path) -> None:
    write_tmqm_directory(tmp_path / "raw")
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(
        f"""
output_root: {tmp_path / "datasets"}
work_root: {tmp_path / "work"}
settings: {{shard_size: 2, workers: 2}}
sources:
  - kind: tmqm
    directory: {tmp_path / "raw"}
"""
    )
    config = read_model(config_path, StructurePreparationConfig)
    report = prepare_structure_datasets(config)
    assert report.n_failed == 0, report.statuses[0].traceback
    output = tmp_path / "datasets" / "tmqm"
    bundle = read_bundle(output)
    table = bundle.table
    assert not bundle.spec.smiles
    assert table["source_key"].tolist() == ["AAAAAA", "BBBBBB", "DDDDDD"]
    assert table["stereoisomer_id"].tolist() == [0, 1, 2]
    assert table["total_charge"].tolist() == [0.0, 1.0, -1.0]
    assert table["Electronic_E"].tolist()[:2] == [0.0, 10.0]
    assert np.isnan(table["HL_Gap"].iloc[2])
    assert table["metal_neighbour_count"].tolist() == [4, 4, 4]
    assert pd.isna(table["tmqm_smiles"].iloc[1])
    assert read_provenance(output).counts.dropped == {
        "min_interatomic_distance": 1,
        "zero_hydrogen": 1,
    }
    assert verify_dataset(output).problems == []
    with pytest.raises(ValueError, match="not in the config"):
        prepare_structure_datasets(config, only=["qm9"])


# ------------------------------------------------------------------- GEOM-style


def test_geom_source_keeps_the_top_conformers_by_weight(tmp_path: Path) -> None:
    folder = tmp_path / "rdkit_folder"
    (folder / "drugs").mkdir(parents=True)
    weights = {"CCO": [0.1, 0.6, 0.3], "CC(=O)[O-]": [1.0], "c1ccccc1N": [0.5, 0.5]}
    summary: dict[str, dict[str, object]] = {}
    for smiles, molecule_weights in weights.items():
        pickle_path = f"drugs/{len(summary)}.pickle"
        conformers = [
            {"rd_mol": embedded_molecule(smiles, seed=seed), "boltzmannweight": weight}
            for seed, weight in enumerate(molecule_weights)
        ]
        with (folder / pickle_path).open("wb") as handle:
            pickle.dump({"conformers": conformers}, handle)
        summary[smiles] = {"pickle_path": pickle_path}
    summary["CCN"] = {"pickle_path": "drugs/missing.pickle"}
    summary["CCC"] = {}  # an entry without a pickle path
    (folder / "summary_drugs.json").write_text(json.dumps(summary))
    source = GeomDrugsSource(directory=folder, conformers_per_molecule=2)
    output = tmp_path / "datasets" / "geom_drugs"
    result = prepare_structures(source, SETTINGS, output, tmp_path / "work")
    table = read_bundle(output).table
    assert table["source_key"].tolist() == [
        "CCO#0",
        "CCO#1",
        "c1ccccc1N#0",
        "c1ccccc1N#1",
    ]
    assert table["boltzmann_weight"].tolist() == [0.6, 0.3, 0.5, 0.5]
    assert table["conformer_rank"].tolist() == [0, 1, 0, 1]
    assert table["molecule_id"].tolist() == [0, 0, 1, 1]
    # The acetate is charged (the GEOM filter keeps neutral molecules only).
    assert result.dropped == {"missing_pickle": 2, SMILES_FILTER: 1}
    # five molecules in the summary, although seven records (five conformers
    # and two missing pickles) were read
    assert read_provenance(output).counts.source_molecules == 5
    assert verify_dataset(output).problems == []


def test_the_production_config_parses() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "dataset_creation"
        / "prepare_structures.yaml"
    )
    config = read_model(path, StructurePreparationConfig)
    assert [source.kind for source in config.sources] == [
        "tmqm",
        "pcqm4mv2",
        "geom_drugs",
    ]


@pytest.mark.parametrize("source", [Pcqm4mv2Source, GeomDrugsSource])
def test_the_drug_like_corpora_drop_radicals(source) -> None:
    """PCQM4Mv2 and GEOM keep closed-shell molecules only (their DFT is closed-shell)."""
    smiles_filter = source.model_fields["smiles_filter"].default_factory()
    assert molecule_passes_filter(Chem.MolFromSmiles("CCN"), smiles_filter)
    assert not molecule_passes_filter(Chem.MolFromSmiles("CC[NH]"), smiles_filter)


def test_the_fragment_gate_counts_fragments_without_sanitising_them() -> None:
    """Only the number of fragments matters; building each one as a sanitised
    molecule used to reject a molecule for a reason the gate does not name."""
    unsanitised = Chem.MolFromSmiles("CC(C)(C)(C)C", sanitize=False)
    assert molecule_passes_filter(unsanitised, FILTER)
    two_fragments = Chem.MolFromSmiles("CCO.CCN")
    assert not molecule_passes_filter(two_fragments, FILTER)
    allowing = FILTER.model_copy(update={"allow_multifragment": True})
    assert molecule_passes_filter(two_fragments, allowing)
