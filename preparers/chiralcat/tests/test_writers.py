"""Output writers: extxyz framing, CSV columns and the run report."""

from __future__ import annotations

import csv
import json

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


def _structure() -> Structure:
    return Structure(
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


def _rejected(with_geometry: bool = True) -> RejectedRecord:
    return RejectedRecord(
        index=11,
        smiles="[Fe].c1ccccc1",
        class_name="planar",
        label=4,
        source_file="chiral_data_3d_aug.pkl",
        stage="repair",
        reason="flagged",
        disposition=BROKEN,
        detail="metal_organometallic: left unrepaired",
        symbols=["Fe", "C"] if with_geometry else [],
        coords=[(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)] if with_geometry else [],
    )


def test_dataset_extxyz_frames_are_well_formed(tmp_path):
    path = tmp_path / "dataset.extxyz"
    write_dataset_extxyz([_structure()], path)
    lines = path.read_text().splitlines()
    assert lines[0] == "2"  # atom count
    assert "index=7" in lines[1]
    assert 'smiles="C[C@H](N)C(=O)O"' in lines[1]
    assert "geometry_quality=repaired" in lines[1]
    assert 'pbc="F F F"' in lines[1]
    assert lines[2].startswith("C ")
    assert len(lines) == 4


def test_rejected_extxyz_skips_records_without_geometry(tmp_path):
    path = tmp_path / "rejected.extxyz"
    written = write_rejected_extxyz([_rejected(False), _rejected(True)], path)
    assert written == 1
    assert "reason=flagged" in path.read_text()


def test_dataset_index_has_the_full_provenance_columns(tmp_path):
    path = tmp_path / "dataset.csv"
    write_dataset_index([_structure()], path)
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == DATASET_COLUMNS
    assert rows[0]["source_file"] == "chiral_data_3d_aug.pkl"
    assert rows[0]["geometry_quality"] == "repaired"
    assert rows[0]["repair_strategy"] == "reembed"
    assert rows[0]["correction"] == "relabel central->achiral"


def test_rejected_index_records_stage_reason_and_disposition(tmp_path):
    path = tmp_path / "rejected.csv"
    write_rejected_index([_rejected()], path)
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == REJECTED_COLUMNS
    assert rows[0]["stage"] == "repair"
    assert rows[0]["reason"] == "flagged"
    assert rows[0]["disposition"] == BROKEN
    assert rows[0]["has_geometry"] == "True"


def test_rejected_index_handles_a_record_with_no_index(tmp_path):
    record = _rejected()
    record.index = None
    path = tmp_path / "rejected.csv"
    write_rejected_index([record], path)
    assert next(iter(csv.DictReader(path.open())))["index"] == ""


def test_run_report_summarises_both_sets(tmp_path):
    result = BuildResult(
        structures=[_structure()],
        rejected=[_rejected()],
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
    assert report["rejected"]["broken"] == 1
    assert report["rejected"]["by_stage_and_reason"] == {"repair:flagged": 1}
    assert report["corrections"]["applied"] == 1
    assert report["stereo_audit"]["uncovered_mislabels"] == ["CCC"]
    assert report["config"]["extraction"]["data_dir"] == "data"


def test_build_result_splits_broken_from_filtered():
    broken = _rejected()
    filtered = _rejected()
    filtered.disposition = FILTERED
    result = BuildResult(rejected=[broken, filtered])
    assert result.rejected_broken == [broken]
    assert result.rejected_filtered == [filtered]
