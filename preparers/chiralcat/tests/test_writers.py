"""Output writers: extxyz framing, CSV columns and the run report."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from chiralcat_dataset.records import (
    BROKEN,
    FILTERED,
    BuildResult,
    CorrectionRecord,
    RejectedRecord,
    Structure,
)
from chiralcat_dataset.writers import (
    DATASET_COLUMNS,
    REJECTED_COLUMNS,
    write_dataset_extxyz,
    write_dataset_index,
    write_rejected_extxyz,
    write_rejected_index,
    write_run_report,
)

STRUCTURE = Structure(
    index=7,
    smiles="C[C@H](N)C(=O)O",
    class_name="central",
    label=1,
    symbols=["C", "O"],
    coords=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)],
    source_file="chiral_data_3d_aug.pkl",
    geometry_quality="repaired",
    repair_strategy="reembed",
    correction="relabel central->achiral",
)
REJECTED = RejectedRecord(
    index=11,
    smiles="[Fe].c1ccccc1",
    class_name="planar",
    label=4,
    source_file="chiral_data_3d_aug.pkl",
    stage="repair",
    reason="flagged",
    disposition=BROKEN,
    detail="metal_organometallic: left unrepaired",
    symbols=["Fe", "C"],
    coords=[(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)],
)


def test_dataset_extxyz_frames_are_well_formed(tmp_path):
    path = tmp_path / "dataset.extxyz"
    write_dataset_extxyz([STRUCTURE], path)
    lines = path.read_text().splitlines()
    assert lines[0] == "2"  # atom count
    for field in ("index=7", 'smiles="C[C@H](N)C(=O)O"', "geometry_quality=repaired"):
        assert field in lines[1]
    assert 'pbc="F F F"' in lines[1]
    assert lines[2].startswith("C ")
    assert len(lines) == 4


def test_rejected_extxyz_skips_records_without_geometry(tmp_path):
    path = tmp_path / "rejected.extxyz"
    without_geometry = replace(REJECTED, symbols=[], coords=[])
    assert write_rejected_extxyz([without_geometry, REJECTED], path) == 1
    assert "reason=flagged" in path.read_text()


@pytest.mark.parametrize(
    ("writer", "record", "columns", "expected"),
    [
        (
            write_dataset_index,
            STRUCTURE,
            DATASET_COLUMNS,
            {
                "source_file": "chiral_data_3d_aug.pkl",
                "geometry_quality": "repaired",
                "repair_strategy": "reembed",
                "correction": "relabel central->achiral",
            },
        ),
        (
            write_rejected_index,
            replace(REJECTED, index=None),
            REJECTED_COLUMNS,
            {
                "index": "",
                "stage": "repair",
                "reason": "flagged",
                "disposition": BROKEN,
                "has_geometry": "True",
            },
        ),
    ],
)
def test_csv_indexes_carry_the_provenance_columns(
    tmp_path, writer, record, columns, expected
):
    path = tmp_path / "index.csv"
    writer([record], path)
    with path.open() as handle:
        [row] = list(csv.DictReader(handle))
    assert list(row) == columns
    assert {key: row[key] for key in expected} == expected


def test_run_report_summarises_both_sets(tmp_path):
    result = BuildResult(
        structures=[STRUCTURE],
        rejected=[REJECTED, replace(REJECTED, disposition=FILTERED)],
        corrections=[
            CorrectionRecord("CCO", "relabel", "central", "achiral", "high", "r", True)
        ],
        uncovered_mislabels=["CCC"],
        stage_counts={"extract": {"kept": 1}},
    )
    path = tmp_path / "run_report.json"
    write_run_report(result, path, {"extraction": {"data_dir": "data"}})
    report = json.loads(path.read_text())

    assert report["dataset"]["structures"] == 1
    assert report["dataset"]["by_class"]["central"] == 1
    assert report["dataset"]["by_geometry_quality"] == {"repaired": 1}
    assert (report["rejected"]["broken"], report["rejected"]["filtered"]) == (1, 1)
    assert report["rejected"]["by_stage_and_reason"] == {"repair:flagged": 2}
    assert report["corrections"]["applied"] == 1
    assert report["stereo_audit"]["uncovered_mislabels"] == ["CCC"]
    assert report["config"]["extraction"]["data_dir"] == "data"
