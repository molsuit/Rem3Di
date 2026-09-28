"""Stage 2: the stereo audit that keeps corrections.yaml honest."""

from __future__ import annotations

import pytest
import yaml
from conftest import data_present
from rdkit import Chem
from rdkit.Chem import AllChem

from chiralcat_dataset.config import PipelineConfig
from chiralcat_dataset.records import Structure
from chiralcat_dataset.validation import (
    UncoveredMislabelError,
    audit_central_stereo,
    validate,
)

SEED = 0xC0FFEE


def _structure(smiles: str, class_name: str = "central") -> Structure:
    from chiralcat_dataset.taxonomy import CLASS_TO_LABEL

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    conformer = mol.GetConformer()
    return Structure(
        index=0,
        smiles=Chem.CanonSmiles(smiles),
        class_name=class_name,
        label=CLASS_TO_LABEL[class_name],
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
        coords=[tuple(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        source_file="test.pkl",
        mol=mol,
    )


def _config(tmp_path, corrections=None, policy="warn") -> PipelineConfig:
    payload = {
        "extraction": {
            "data_dir": "data",
            "sources": [{"kind": "typed", "path": "typed.pkl"}],
        },
        "validation": {
            "corrections_file": "corrections.yaml" if corrections else None,
            "audit_central_stereo": True,
            "on_uncovered_mislabel": policy,
        },
    }
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(payload))
    if corrections:
        (tmp_path / "corrections.yaml").write_text(
            yaml.safe_dump({"corrections": corrections})
        )
    return PipelineConfig.from_yaml(tmp_path / "pipeline.yaml")


# --------------------------------------------------------------------------- #
# The audit itself
# --------------------------------------------------------------------------- #


def test_audit_only_looks_at_the_central_class():
    structures = [_structure("CCO", "achiral"), _structure("C[C@H](N)C(=O)O")]
    records = audit_central_stereo(structures)
    assert len(records) == 1
    assert records[0].smiles == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_audit_finds_a_real_stereocentre():
    records = audit_central_stereo([_structure("C[C@H](N)C(=O)O")])
    assert records[0].assigned_rs_3d == 1
    assert records[0].has_stereocenter is True


def test_audit_reports_no_stereocentre_for_a_mislabelled_molecule():
    records = audit_central_stereo([_structure("CCO")])
    assert records[0].assigned_rs_3d == 0
    assert records[0].has_stereocenter is False


def test_audit_records_a_perception_failure_rather_than_raising():
    structure = _structure("C[C@H](N)C(=O)O")
    structure.mol = None  # simulate a molecule the extraction could not carry
    records = audit_central_stereo([structure])
    assert records[0].error
    assert records[0].assigned_rs_3d == -1


# --------------------------------------------------------------------------- #
# Policy on uncovered mislabels
# --------------------------------------------------------------------------- #


def test_suspected_mislabel_is_reported_when_uncovered(tmp_path):
    config = _config(tmp_path)
    output = validate(config, [_structure("CCO")])
    assert output.uncovered == [Chem.CanonSmiles("CCO")]
    assert output.counts["no_stereocenter"] == 1


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
    output = validate(config, [_structure("CCO")])
    assert output.uncovered == ["CCO"]


def test_error_policy_aborts_the_build(tmp_path):
    config = _config(tmp_path, policy="error")
    with pytest.raises(UncoveredMislabelError, match="no 3D stereocentre"):
        validate(config, [_structure("CCO")])


def test_audit_can_be_switched_off(tmp_path):
    config = _config(tmp_path)
    config.validation.audit_central_stereo = False
    output = validate(config, [_structure("CCO")])
    assert output.audit == []
    assert output.uncovered == []


def test_keep_annotations_still_apply_when_the_audit_is_off(tmp_path):
    """A keep records a review decision, not an artefact of the audit."""
    config = _config(
        tmp_path,
        corrections=[{"kind": "keep", "smiles": "CCO", "reason": "reviewed"}],
    )
    config.validation.audit_central_stereo = False
    structure = _structure("CCO")
    output = validate(config, [structure])
    assert output.audit == []
    assert "keep central (reviewed)" in structure.correction
    assert output.counts["kept_by_review"] == 1


def test_molecule_with_a_stereocentre_never_counts_as_a_mislabel(tmp_path):
    config = _config(tmp_path)
    output = validate(config, [_structure("C[C@H](N)C(=O)O")])
    assert output.uncovered == []
    assert output.counts["with_stereocenter"] == 1


# --------------------------------------------------------------------------- #
# The `keep` decision
# --------------------------------------------------------------------------- #


def test_keep_correction_silences_a_reviewed_molecule(tmp_path):
    """A reviewed-and-kept molecule stops being reported as a finding."""
    config = _config(
        tmp_path,
        corrections=[
            {
                "kind": "keep",
                "smiles": "CCO",
                "reason": "P(III) lone pair; genuinely chiral, unperceived in 3D",
                "confidence": "high",
            }
        ],
    )
    output = validate(config, [_structure("CCO")])
    assert output.uncovered == []
    # Still counted as having no 3D stereocentre - only the alarm is off.
    assert output.counts["no_stereocenter"] == 1
    assert output.counts["kept_by_review"] == 1


def test_keep_correction_annotates_the_structure_without_changing_its_class(tmp_path):
    config = _config(
        tmp_path,
        corrections=[{"kind": "keep", "smiles": "CCO", "reason": "reviewed"}],
    )
    structure = _structure("CCO")
    validate(config, [structure])
    assert structure.class_name == "central"  # unchanged
    assert structure.label == 1
    assert "keep central (reviewed)" in structure.correction


def test_keep_correction_is_recorded_as_a_matched_correction(tmp_path):
    config = _config(
        tmp_path,
        corrections=[{"kind": "keep", "smiles": "CCO", "reason": "reviewed"}],
    )
    output = validate(config, [_structure("CCO")])
    assert len(output.corrections) == 1
    assert output.corrections[0].action == "keep"
    assert output.corrections[0].matched is True
    assert output.corrections[0].from_class == output.corrections[0].to_class


def test_stale_keep_correction_is_surfaced_as_unmatched(tmp_path):
    config = _config(
        tmp_path,
        corrections=[{"kind": "keep", "smiles": "CCCCCCCC", "reason": "stale"}],
    )
    output = validate(config, [_structure("C[C@H](N)C(=O)O")])
    unmatched = [c for c in output.corrections if not c.matched]
    assert len(unmatched) == 1
    assert unmatched[0].action == "keep"


def test_keep_matches_the_standardized_smiles_the_audit_reports(tmp_path):
    """The audit keys on the shipped SMILES, so a keep entry must too."""
    from rdkit import Chem

    structure = _structure("CCO")
    assert structure.smiles == Chem.CanonSmiles("CCO")
    # An equivalent but non-canonical spelling still matches.
    config = _config(
        tmp_path, corrections=[{"kind": "keep", "smiles": "OCC", "reason": "reviewed"}]
    )
    output = validate(config, [structure])
    assert output.uncovered == []


def test_keep_correction_with_unparseable_smiles_raises(tmp_path):
    config = _config(tmp_path, corrections=[{"kind": "keep", "smiles": "!!!nonsense!!!"}])
    with pytest.raises(ValueError, match="does not parse"):
        validate(config, [_structure("CCO")])


# --------------------------------------------------------------------------- #
# Against the real data
# --------------------------------------------------------------------------- #

KEPT_BY_REVIEW = 28


@data_present
def test_real_build_has_no_uncovered_mislabels(build):
    """corrections.yaml must still cover every mislabel the audit can find."""
    assert build.uncovered_mislabels == [], (
        "the stereo audit found central molecules with no 3D stereocentre that "
        f"corrections.yaml does not cover: {build.uncovered_mislabels}"
    )


@data_present
def test_real_keep_corrections_all_match(build):
    keeps = [c for c in build.corrections if c.action == "keep"]
    assert len(keeps) == KEPT_BY_REVIEW
    stale = [c.smiles for c in keeps if not c.matched]
    assert stale == [], f"corrections.yaml has stale keep entries: {stale}"
