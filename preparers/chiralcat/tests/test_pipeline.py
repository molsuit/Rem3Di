"""End-to-end: the orchestrator over the real source pickles."""

from __future__ import annotations

import csv

import numpy as np
from conftest import data_present

from chiralcat_dataset.chemistry import find_clashes
from chiralcat_dataset.pipeline import write_outputs
from chiralcat_dataset.taxonomy import CLASS_TO_LABEL

FROZEN_REBUILDS = 28
FILTER_REASONS = {
    "duplicate",
    "duplicate_after_standardize",
    "dropped_type",
    "correction_delete",
    "dropped_duplicate",
}


@data_present
def test_every_shipped_structure_is_well_formed(build):
    assert set(build.stage_counts) == {"extract", "validate", "repair", "organometallic"}
    for structure in build.structures:
        assert structure.n_atoms == len(structure.coords) > 0
        assert structure.label == CLASS_TO_LABEL[structure.class_name]
        assert structure.source_file, "every structure records its source pickle"
        assert structure.geometry_quality in {"ok", "repaired", "rebuilt_architector"}
        assert "*" not in structure.symbols
    assert len({s.index for s in build.structures}) == len(build.structures)
    assert len({s.smiles for s in build.structures}) == len(build.structures)


@data_present
def test_the_usable_dataset_is_clash_free(build):
    """The point of the repair and rebuild stages: nothing that ships still clashes."""
    clashing = [
        structure.index
        for structure in build.structures
        if find_clashes(structure.symbols, np.array(structure.coords))
    ]
    assert clashing == [], f"{len(clashing)} shipped structures still clash"


@data_present
def test_hydrogens_sit_within_a_bond_length_of_a_heavy_atom(build):
    bad = 0
    # Spot-check the chiral classes; the achiral bulk is the same code path.
    for structure in build.structures:
        if structure.class_name == "achiral":
            continue
        coords = np.array(structure.coords)
        is_hydrogen = np.array([symbol == "H" for symbol in structure.symbols])
        heavy = coords[~is_hydrogen]
        if not len(heavy):
            continue
        for hydrogen in coords[is_hydrogen]:
            if float(np.min(np.linalg.norm(heavy - hydrogen, axis=1))) > 1.4:
                bad += 1
    assert bad == 0, f"{bad} hydrogens with no heavy atom within bonding distance"


@data_present
def test_rejections_are_disjoint_from_the_dataset_and_classified(build):
    kept = {s.index for s in build.structures}
    assert kept.isdisjoint(r.index for r in build.rejected if r.index is not None)
    for record in build.rejected:
        assert record.stage in {"extract", "validate", "repair", "organometallic"}
        assert record.reason and record.source_file
        assert record.disposition in {"broken", "filtered"}
    # A broken record is never something the pipeline merely chose to drop.
    assert not FILTER_REASONS & {record.reason for record in build.rejected_broken}


@data_present
def test_every_frozen_rebuild_ships_as_a_planar_metal_complex(build):
    rebuilt = [s for s in build.structures if s.geometry_quality == "rebuilt_architector"]
    assert len(rebuilt) == FROZEN_REBUILDS
    for structure in rebuilt:
        assert structure.class_name == "planar"
        assert structure.repair_strategy == "architector"
        assert any(symbol in {"Fe", "Cr", "Mn"} for symbol in structure.symbols)


@data_present
def test_write_outputs_round_trips_both_datasets(config, build, tmp_path):
    config = config.model_copy(deep=True)
    config.output.directory = str(tmp_path)
    config.base_dir = tmp_path
    paths = write_outputs(config, build)

    for path in paths.values():
        assert path.is_file(), f"{path} was not written"
    with paths["dataset_index"].open() as handle:
        assert len(list(csv.DictReader(handle))) == len(build.structures)
    with paths["rejected_index"].open() as handle:
        assert len(list(csv.DictReader(handle))) == len(build.rejected)

    # One extxyz frame per structure: the atom-count lines must add up.
    text = paths["dataset_extxyz"].read_text().splitlines()
    assert len(text) == sum(s.n_atoms + 2 for s in build.structures)
