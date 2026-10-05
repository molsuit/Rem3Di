"""Stage 1: extraction, deduplication and the manual label corrections."""

from __future__ import annotations

import pytest
from conftest import (
    ALANINE,
    FIXED_ACHIRAL_SOURCE,
    TYPED_SOURCE,
    data_present,
    write_config,
    write_pickle,
)

from chiralcat_dataset.config import PipelineConfig
from chiralcat_dataset.extract import ExtractionOutput, extract
from chiralcat_dataset.records import BROKEN, FILTERED
from chiralcat_dataset.sources import ChecksumMismatchError, sha256_of_file


def _extract(tmp_path, pickles: dict[str, dict], sources, corrections=None):
    for name, payload in pickles.items():
        write_pickle(tmp_path / "data" / name, payload)
    path = write_config(tmp_path, sources=sources, corrections=corrections)
    return extract(PipelineConfig.from_yaml(path))


def test_extracts_labelled_structures_with_hydrogens(tmp_path):
    output = _extract(
        tmp_path,
        {
            "typed.pkl": {"SMILES": [ALANINE], "chiral type": ["center"]},
            "fixed.pkl": {"SMILES": ["CCO", "CCC"]},
        },
        [TYPED_SOURCE, FIXED_ACHIRAL_SOURCE],
    )
    assert [s.index for s in output.structures] == [0, 1, 2]
    structure = output.structures[0]
    assert structure.class_name == "central"  # 'center' synonym normalised
    assert structure.label == 1
    assert "H" in structure.symbols
    assert len(structure.coords) == structure.n_atoms
    assert structure.source_file == "typed.pkl"


def test_a_source_pinned_to_another_sha256_is_refused(tmp_path):
    write_pickle(tmp_path / "data" / "fixed.pkl", {"SMILES": ["CCO"]})

    def run(sha256: str) -> ExtractionOutput:
        source = {**FIXED_ACHIRAL_SOURCE, "sha256": sha256}
        return extract(PipelineConfig.from_yaml(write_config(tmp_path, sources=[source])))

    with pytest.raises(ChecksumMismatchError):
        run("0" * 64)
    assert len(run(sha256_of_file(tmp_path / "data" / "fixed.pkl")).structures) == 1


DELETE_ALANINE = [{"kind": "delete", "smiles": ALANINE, "reason": "ambiguous"}]


@pytest.mark.parametrize(
    ("payload", "corrections", "reason", "disposition"),
    [
        ({"SMILES": ["CCO", "CCO"]}, None, "duplicate", FILTERED),
        ({"SMILES": ["CCO", ALANINE], "chiral type": ["unknown", "center"]}, None,
         "dropped_type", FILTERED),
        ({"SMILES": [ALANINE, "CCO"], "chiral type": ["center"] * 2}, DELETE_ALANINE,
         "correction_delete", FILTERED),
        ({"SMILES": ["not a molecule", "CCC"], "mol": ["CCO", "CCC"]}, None,
         "bad_smiles", BROKEN),
        ({"SMILES": ["CCO", "CCC"], "mol": [None, "CCC"]}, None, "no_mol", BROKEN),
    ],
)  # fmt: skip
def test_each_rejection_is_recorded_with_its_disposition(
    tmp_path, payload, corrections, reason, disposition
):
    source = TYPED_SOURCE if "chiral type" in payload else FIXED_ACHIRAL_SOURCE
    output = _extract(tmp_path, {source["path"]: payload}, [source], corrections)
    assert len(output.structures) == 1
    assert [(r.reason, r.disposition) for r in output.rejected] == [(reason, disposition)]


@pytest.mark.parametrize(
    ("payload", "corrections", "error", "match"),
    [
        (None, None, FileNotFoundError, "Source file not found"),
        ({"SMILES": ["CCO", "CCC"], "mol": ["CCO"]}, None, ValueError, "length mismatch"),
        (
            {"SMILES": ["CCO"]},
            [{"kind": "delete", "smiles": "!!!not a molecule!!!"}],
            ValueError,
            "does not parse",
        ),
    ],
)
def test_unusable_input_aborts_the_extraction(
    tmp_path, payload, corrections, error, match
):
    pickles = {} if payload is None else {"fixed.pkl": payload}
    with pytest.raises(error, match=match):
        _extract(tmp_path, pickles, [FIXED_ACHIRAL_SOURCE], corrections)


def test_relabel_correction_changes_class_and_is_recorded(tmp_path):
    output = _extract(
        tmp_path,
        {"typed.pkl": {"SMILES": ["OC1(c2ccccc2)CCNCC1"], "chiral type": ["center"]}},
        [TYPED_SOURCE],
        corrections=[
            {
                "kind": "relabel",
                "smiles": "OC1(c2ccccc2)CCNCC1",
                "to_class": "achiral",
                "reason": "not stereogenic",
            }
        ],
    )
    assert output.structures[0].class_name == "achiral"
    assert output.structures[0].label == 0
    assert "relabel central->achiral" in output.structures[0].correction
    assert output.corrections[0].matched is True


def test_unmatched_correction_is_surfaced(tmp_path):
    output = _extract(
        tmp_path,
        {"typed.pkl": {"SMILES": ["CCO"], "chiral type": ["center"]}},
        [TYPED_SOURCE],
        corrections=[{"kind": "relabel", "smiles": "CCCCCCCC", "to_class": "achiral"}],
    )
    assert [c.matched for c in output.corrections] == [False]
    assert output.counts["correction_unmatched"] == 1


# The decisions the shipped corrections.yaml is expected to realise.
EXPECTED_CORRECTIONS = {"relabel": 6, "delete": 6, "keep": 28}


@data_present
def test_real_corrections_all_match(build):
    """Extraction applies relabel and delete, validation applies keep."""
    stale = [c.smiles for c in build.corrections if not c.matched]
    assert stale == [], f"corrections.yaml has stale entries: {stale}"
    actions = [c.action for c in build.corrections]
    assert {action: actions.count(action) for action in set(actions)} == (
        EXPECTED_CORRECTIONS
    )
