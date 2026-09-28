"""End-to-end: the orchestrator over the real source pickles."""

from __future__ import annotations

import csv

import numpy as np
from conftest import data_present

from chiralcat_dataset.chemistry import find_clashes
from chiralcat_dataset.pipeline import write_outputs
from chiralcat_dataset.taxonomy import CLASS_ORDER, CLASS_TO_LABEL


@data_present
def test_every_structure_is_internally_consistent(build):
    for structure in build.structures:
        assert structure.n_atoms > 0
        assert len(structure.coords) == structure.n_atoms
        assert structure.label == CLASS_TO_LABEL[structure.class_name]
        assert structure.class_name in CLASS_ORDER
        assert structure.source_file, "every structure records its source pickle"
        assert structure.geometry_quality in {"ok", "repaired", "rebuilt_architector"}


@data_present
def test_indices_and_smiles_are_unique(build):
    indices = [s.index for s in build.structures]
    smiles = [s.smiles for s in build.structures]
    assert len(set(indices)) == len(indices)
    assert len(set(smiles)) == len(smiles)


@data_present
def test_the_usable_dataset_is_clash_free(build):
    """The point of the repair stage: nothing that ships still clashes."""
    clashing = [
        structure.index
        for structure in build.structures
        if find_clashes(structure.symbols, np.array(structure.coords))
    ]
    assert clashing == [], f"{len(clashing)} shipped structures still clash"


@data_present
def test_rejected_and_kept_sets_are_disjoint(build):
    kept = {s.index for s in build.structures}
    rejected = {r.index for r in build.rejected if r.index is not None}
    assert kept.isdisjoint(rejected)


@data_present
def test_every_rejection_names_a_stage_reason_and_disposition(build):
    for record in build.rejected:
        assert record.stage in {"extract", "validate", "repair", "organometallic"}
        assert record.reason
        assert record.disposition in {"broken", "filtered"}
        assert record.source_file


@data_present
def test_broken_records_are_the_ones_that_could_not_be_fixed(build):
    """A broken record is never something the pipeline merely chose to drop."""
    for record in build.rejected_broken:
        assert record.reason not in {
            "duplicate",
            "duplicate_after_standardize",
            "dropped_type",
            "correction_delete",
            "dropped_duplicate",
        }


@data_present
def test_hydrogens_sit_within_a_bond_length_of_a_heavy_atom(build):
    bad = 0
    # Spot-check the chiral classes; the achiral bulk is the same code path.
    for structure in build.structures:
        if structure.class_name == "achiral":
            continue
        coords = np.array(structure.coords)
        heavy = coords[[i for i, s in enumerate(structure.symbols) if s != "H"]]
        if not len(heavy):
            continue
        for i, symbol in enumerate(structure.symbols):
            if symbol != "H":
                continue
            if float(np.min(np.linalg.norm(heavy - coords[i], axis=1))) > 1.4:
                bad += 1
    assert bad == 0, f"{bad} hydrogens with no heavy atom within bonding distance"


@data_present
def test_no_structure_carries_a_dummy_atom(build):
    assert not any("*" in s.symbols for s in build.structures)


@data_present
def test_organometallic_rebuilds_are_marked_and_planar(build):
    rebuilt = [s for s in build.structures if s.geometry_quality == "rebuilt_architector"]
    assert rebuilt, "the rebuild stage produced nothing"
    for structure in rebuilt:
        assert structure.class_name == "planar"
        assert structure.repair_strategy == "architector"


@data_present
def test_rebuilt_structures_are_clash_free_and_complete(build):
    """The rebuild's own validation gate must hold in the shipped dataset."""
    rebuilt = [s for s in build.structures if s.geometry_quality == "rebuilt_architector"]
    for structure in rebuilt:
        assert find_clashes(structure.symbols, np.array(structure.coords)) == []
        assert any(symbol in {"Fe", "Cr", "Mn"} for symbol in structure.symbols)


@data_present
def test_rebuild_moves_structures_out_of_the_rejected_set(build):
    """Everything the rebuild rescued must no longer be listed as rejected."""
    rebuilt_indices = {
        s.index for s in build.structures if s.geometry_quality == "rebuilt_architector"
    }
    rejected_indices = {r.index for r in build.rejected}
    assert rebuilt_indices.isdisjoint(rejected_indices)


@data_present
def test_stage_counts_are_recorded_for_every_stage(build):
    assert set(build.stage_counts) == {
        "extract",
        "validate",
        "repair",
        "organometallic",
    }


@data_present
def test_write_outputs_round_trips_both_datasets(config, build, tmp_path):
    config = config.model_copy(deep=True)
    config.output.directory = str(tmp_path)
    config.base_dir = tmp_path
    paths = write_outputs(config, build)

    for path in paths.values():
        assert path.is_file(), f"{path} was not written"

    dataset_rows = list(csv.DictReader(paths["dataset_index"].open()))
    assert len(dataset_rows) == len(build.structures)

    rejected_rows = list(csv.DictReader(paths["rejected_index"].open()))
    assert len(rejected_rows) == len(build.rejected)

    # One extxyz frame per structure: the atom-count lines must add up.
    text = paths["dataset_extxyz"].read_text().splitlines()
    total_lines = sum(s.n_atoms + 2 for s in build.structures)
    assert len(text) == total_lines
