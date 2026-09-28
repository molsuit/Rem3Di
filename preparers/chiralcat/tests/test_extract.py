"""Stage 1: extraction, deduplication and the manual label corrections."""

from __future__ import annotations

import pickle

import pytest
from conftest import data_present
from rdkit import Chem
from rdkit.Chem import AllChem

from chiralcat_dataset.config import PipelineConfig
from chiralcat_dataset.extract import extract
from chiralcat_dataset.records import BROKEN, FILTERED
from chiralcat_dataset.sources import ChecksumMismatchError, sha256_of_file

SEED = 0xC0FFEE

# The corrections the shipped corrections.yaml is expected to realise.
RELABELED_TO_ACHIRAL = 6
DELETED = 6


def _mol_with_conformer(smiles: str, *, keep_hydrogens: bool = False) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    return mol if keep_hydrogens else Chem.RemoveHs(mol)


def _write_pickle(path, smiles_list, types=None):
    payload = {
        "SMILES": list(smiles_list),
        "mol": [_mol_with_conformer(s) for s in smiles_list],
    }
    if types is not None:
        payload["chiral type"] = list(types)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _config(tmp_path, sources, corrections=None) -> PipelineConfig:
    import yaml

    payload = {
        "extraction": {"data_dir": "data", "sources": sources},
        "validation": {
            "corrections_file": "corrections.yaml" if corrections else None,
            "audit_central_stereo": False,
        },
    }
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(payload))
    if corrections:
        (tmp_path / "corrections.yaml").write_text(
            yaml.safe_dump({"corrections": corrections})
        )
    return PipelineConfig.from_yaml(tmp_path / "pipeline.yaml")


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    return directory


# --------------------------------------------------------------------------- #
# Core extraction behaviour
# --------------------------------------------------------------------------- #


def test_extracts_labelled_structures_with_hydrogens(tmp_path, data_dir):
    _write_pickle(data_dir / "typed.pkl", ["C[C@H](N)C(=O)O"], ["center"])
    config = _config(tmp_path, [{"kind": "typed", "path": "typed.pkl"}])

    output = extract(config)
    assert len(output.structures) == 1
    structure = output.structures[0]
    assert structure.class_name == "central"  # 'center' synonym normalised
    assert structure.label == 1
    assert "H" in structure.symbols
    assert len(structure.coords) == structure.n_atoms
    assert structure.source_file == "typed.pkl"


def test_indices_are_contiguous_from_zero(tmp_path, data_dir):
    _write_pickle(data_dir / "fixed.pkl", ["CCO", "CCC", "CCCC"])
    config = _config(
        tmp_path, [{"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}]
    )
    output = extract(config)
    assert [s.index for s in output.structures] == [0, 1, 2]


def test_a_source_pinned_to_another_sha256_is_refused(tmp_path, data_dir):
    _write_pickle(data_dir / "fixed.pkl", ["CCO"])
    source = {"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}
    with pytest.raises(ChecksumMismatchError):
        extract(_config(tmp_path, [{**source, "sha256": "0" * 64}]))
    pinned = {**source, "sha256": sha256_of_file(data_dir / "fixed.pkl")}
    assert len(extract(_config(tmp_path, [pinned])).structures) == 1


def test_duplicate_smiles_is_filtered_not_broken(tmp_path, data_dir):
    _write_pickle(data_dir / "fixed.pkl", ["CCO", "CCO"])
    config = _config(
        tmp_path, [{"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}]
    )
    output = extract(config)
    assert len(output.structures) == 1
    assert len(output.rejected) == 1
    assert output.rejected[0].reason == "duplicate"
    assert output.rejected[0].disposition == FILTERED


def test_dropped_type_is_filtered(tmp_path, data_dir):
    _write_pickle(
        data_dir / "typed.pkl", ["CCO", "C[C@H](N)C(=O)O"], ["unknown", "center"]
    )
    config = _config(tmp_path, [{"kind": "typed", "path": "typed.pkl"}])
    output = extract(config)
    assert len(output.structures) == 1
    assert [r.reason for r in output.rejected] == ["dropped_type"]
    assert output.rejected[0].disposition == FILTERED


def test_unparseable_smiles_is_broken(tmp_path, data_dir):
    payload = {"SMILES": ["not a molecule"], "mol": [_mol_with_conformer("CCO")]}
    with open(data_dir / "fixed.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    config = _config(
        tmp_path, [{"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}]
    )
    output = extract(config)
    assert output.structures == []
    assert output.rejected[0].reason == "bad_smiles"
    assert output.rejected[0].disposition == BROKEN


def test_missing_mol_is_broken(tmp_path, data_dir):
    payload = {"SMILES": ["CCO"], "mol": [None]}
    with open(data_dir / "fixed.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    config = _config(
        tmp_path, [{"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}]
    )
    output = extract(config)
    assert output.rejected[0].reason == "no_mol"
    assert output.rejected[0].disposition == BROKEN


def test_missing_source_file_raises(tmp_path, data_dir):
    config = _config(
        tmp_path, [{"kind": "fixed_label", "path": "absent.pkl", "label": "achiral"}]
    )
    with pytest.raises(FileNotFoundError):
        extract(config)


def test_length_mismatch_between_parallel_arrays_raises(tmp_path, data_dir):
    payload = {"SMILES": ["CCO", "CCC"], "mol": [_mol_with_conformer("CCO")]}
    with open(data_dir / "fixed.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    config = _config(
        tmp_path, [{"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}]
    )
    with pytest.raises(ValueError, match="length mismatch"):
        extract(config)


# --------------------------------------------------------------------------- #
# Label corrections
# --------------------------------------------------------------------------- #


def test_relabel_correction_changes_class_and_is_recorded(tmp_path, data_dir):
    _write_pickle(data_dir / "typed.pkl", ["OC1(c2ccccc2)CCNCC1"], ["center"])
    config = _config(
        tmp_path,
        [{"kind": "typed", "path": "typed.pkl"}],
        corrections=[
            {
                "kind": "relabel",
                "smiles": "OC1(c2ccccc2)CCNCC1",
                "to_class": "achiral",
                "reason": "not stereogenic",
            }
        ],
    )
    output = extract(config)
    assert output.structures[0].class_name == "achiral"
    assert output.structures[0].label == 0
    assert "relabel central->achiral" in output.structures[0].correction
    assert output.corrections[0].matched is True


def test_delete_correction_removes_the_molecule(tmp_path, data_dir):
    _write_pickle(data_dir / "typed.pkl", ["C[C@H](N)C(=O)O"], ["center"])
    config = _config(
        tmp_path,
        [{"kind": "typed", "path": "typed.pkl"}],
        corrections=[
            {"kind": "delete", "smiles": "C[C@H](N)C(=O)O", "reason": "ambiguous"}
        ],
    )
    output = extract(config)
    assert output.structures == []
    assert output.rejected[0].reason == "correction_delete"
    assert output.rejected[0].disposition == FILTERED


def test_unmatched_correction_is_surfaced(tmp_path, data_dir):
    _write_pickle(data_dir / "typed.pkl", ["CCO"], ["center"])
    config = _config(
        tmp_path,
        [{"kind": "typed", "path": "typed.pkl"}],
        corrections=[{"kind": "relabel", "smiles": "CCCCCCCC", "to_class": "achiral"}],
    )
    output = extract(config)
    unmatched = [c for c in output.corrections if not c.matched]
    assert len(unmatched) == 1
    assert output.counts["correction_unmatched"] == 1


def test_correction_with_unparseable_smiles_raises(tmp_path, data_dir):
    _write_pickle(data_dir / "typed.pkl", ["CCO"], ["center"])
    config = _config(
        tmp_path,
        [{"kind": "typed", "path": "typed.pkl"}],
        corrections=[{"kind": "delete", "smiles": "!!!not a molecule!!!"}],
    )
    with pytest.raises(ValueError, match="does not parse"):
        extract(config)


# --------------------------------------------------------------------------- #
# Against the real pickles
# --------------------------------------------------------------------------- #


@data_present
def test_real_extraction_is_internally_consistent(fast_config):
    output = extract(fast_config)
    smiles = [s.smiles for s in output.structures]
    assert len(set(smiles)) == len(smiles)  # no molecule appears twice
    for structure in output.structures:
        assert structure.n_atoms > 0
        assert len(structure.coords) == structure.n_atoms
        assert structure.source_file  # provenance is always populated
        assert structure.mol is not None  # carried for the validation stage


@data_present
def test_real_corrections_all_match(fast_config):
    output = extract(fast_config)
    unmatched = [c.smiles for c in output.corrections if not c.matched]
    assert unmatched == [], f"corrections.yaml has stale entries: {unmatched}"
    assert sum(1 for c in output.corrections if c.action == "relabel") == (
        RELABELED_TO_ACHIRAL
    )
    assert sum(1 for c in output.corrections if c.action == "delete") == DELETED
    # `keep` is applied by the validation stage, not here.
    assert not any(c.action == "keep" for c in output.corrections)
