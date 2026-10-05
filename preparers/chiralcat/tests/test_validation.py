"""Stage 2: the stereo audit that keeps corrections.yaml honest."""

from __future__ import annotations

import pytest
from conftest import ALANINE, data_present, make_structure, write_config

from chiralcat_dataset.config import PipelineConfig
from chiralcat_dataset.validation import (
    UncoveredMislabelError,
    audit_central_stereo,
    validate,
)


def _config(tmp_path, corrections=None, policy="warn", audit=True) -> PipelineConfig:
    validation = {"audit_central_stereo": audit, "on_uncovered_mislabel": policy}
    path = write_config(tmp_path, corrections=corrections, validation=validation)
    return PipelineConfig.from_yaml(path)


def test_audit_reads_stereocentres_of_the_central_class_only():
    structures = [
        make_structure("CCO", "achiral"),
        make_structure(ALANINE),
        make_structure("CCO"),  # a mislabelled central molecule
    ]
    records = audit_central_stereo(structures)
    assert [(r.smiles, r.assigned_rs_3d, r.has_stereocenter) for r in records] == [
        (ALANINE, 1, True),
        ("CCO", 0, False),
    ]


def test_audit_records_a_perception_failure_rather_than_raising():
    structure = make_structure(ALANINE)
    structure.mol = None  # simulate a molecule the extraction could not carry
    records = audit_central_stereo([structure])
    assert records[0].error
    assert records[0].assigned_rs_3d == -1


def test_only_central_molecules_without_a_stereocentre_are_reported(tmp_path):
    output = validate(_config(tmp_path), [make_structure("CCO"), make_structure(ALANINE)])
    assert output.uncovered == ["CCO"]
    assert output.counts["no_stereocenter"] == 1
    assert output.counts["with_stereocenter"] == 1


def test_relabel_entry_does_not_silence_the_audit(tmp_path):
    """Only `keep` silences a finding.

    A `relabel` moves the molecule out of the central class during extraction,
    so it should never reach the audit as central. If one still does, the label
    did not take effect and the audit should say so rather than stay quiet.
    """
    config = _config(
        tmp_path,
        corrections=[{"kind": "relabel", "smiles": "CCO", "to_class": "achiral"}],
    )
    assert validate(config, [make_structure("CCO")]).uncovered == ["CCO"]


def test_error_policy_aborts_the_build(tmp_path):
    with pytest.raises(UncoveredMislabelError, match="no 3D stereocentre"):
        validate(_config(tmp_path, policy="error"), [make_structure("CCO")])


def test_keep_correction_silences_and_annotates_a_reviewed_molecule(tmp_path):
    """A reviewed-and-kept molecule stops being a finding but keeps its class.

    The audit keys on the shipped canonical SMILES, so an equivalent but
    non-canonical spelling of the keep entry must still match.
    """
    config = _config(
        tmp_path,
        corrections=[
            {
                "kind": "keep",
                "smiles": "OCC",
                "reason": "reviewed",
                "confidence": "high",
            }
        ],
    )
    structure = make_structure("CCO")
    output = validate(config, [structure])
    assert output.uncovered == []
    # Still counted as having no 3D stereocentre - only the alarm is off.
    assert output.counts["no_stereocenter"] == 1
    assert output.counts["kept_by_review"] == 1
    assert (structure.class_name, structure.label) == ("central", 1)
    assert "keep central (reviewed)" in structure.correction
    [correction] = output.corrections
    assert (correction.action, correction.matched) == ("keep", True)
    assert correction.from_class == correction.to_class


def test_keep_annotations_still_apply_when_the_audit_is_off(tmp_path):
    """A keep records a review decision, not an artefact of the audit."""
    config = _config(
        tmp_path,
        corrections=[{"kind": "keep", "smiles": "CCO", "reason": "reviewed"}],
        audit=False,
    )
    kept = make_structure("CCO")
    output = validate(config, [kept, make_structure("CCC")])
    assert output.audit == []
    assert output.uncovered == []  # CCC would be a finding with the audit on
    assert "keep central (reviewed)" in kept.correction
    assert output.counts["kept_by_review"] == 1


def test_stale_keep_correction_is_surfaced_as_unmatched(tmp_path):
    config = _config(
        tmp_path,
        corrections=[{"kind": "keep", "smiles": "CCCCCCCC", "reason": "stale"}],
    )
    output = validate(config, [make_structure(ALANINE)])
    assert [(c.action, c.matched) for c in output.corrections] == [("keep", False)]


def test_keep_correction_with_unparseable_smiles_raises(tmp_path):
    config = _config(tmp_path, corrections=[{"kind": "keep", "smiles": "!!!nonsense!!!"}])
    with pytest.raises(ValueError, match="does not parse"):
        validate(config, [make_structure("CCO")])


@data_present
def test_real_build_has_no_uncovered_mislabels(build):
    """corrections.yaml must still cover every mislabel the audit can find."""
    assert build.uncovered_mislabels == [], (
        "the stereo audit found central molecules with no 3D stereocentre that "
        f"corrections.yaml does not cover: {build.uncovered_mislabels}"
    )
