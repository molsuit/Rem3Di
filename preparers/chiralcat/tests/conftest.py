"""Shared fixtures and molecule helpers for the ingestion-pipeline tests.

The integration tests run the real pipeline over the real pickles, which live
in the gitignored ``benchmark_data/raw/chiralcat`` of the Rem3Di checkout.
Tests that need the data are skipped when the pickles are absent. The full
build takes about a minute and a half, dominated by re-embedding in the repair
stage, so it runs once per session.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from chiralcat_dataset import PipelineConfig, build_dataset
from chiralcat_dataset.records import Structure
from chiralcat_dataset.taxonomy import CLASS_TO_LABEL

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "pipeline.yaml"
SEED = 0xC0FFEE
ALANINE = "C[C@H](N)C(=O)O"
FIXED_ACHIRAL_SOURCE = {"kind": "fixed_label", "path": "fixed.pkl", "label": "achiral"}
TYPED_SOURCE = {"kind": "typed", "path": "typed.pkl"}

data_present = pytest.mark.skipif(
    not (PipelineConfig.from_yaml(CONFIG_PATH).data_dir / "chiral_for_no.pkl").is_file(),
    reason="ChiralCat source pickles not present",
)


def embed(smiles: str) -> Chem.Mol:
    """The molecule with explicit hydrogens and one seeded conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=SEED)
    return mol


def coordinates(mol: Chem.Mol) -> np.ndarray:
    conformer = mol.GetConformer()
    return np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )


def make_structure(smiles: str, class_name: str = "central", index: int = 0) -> Structure:
    """An embedded structure as the extraction stage would hand it on."""
    mol = embed(smiles)
    return Structure(
        index=index,
        smiles=Chem.CanonSmiles(smiles),
        class_name=class_name,
        label=CLASS_TO_LABEL[class_name],
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
        coords=[(x, y, z) for x, y, z in coordinates(mol).tolist()],
        source_file="test.pkl",
        mol=mol,
    )


def write_pickle(path: Path, payload: dict) -> None:
    """Write a source pickle; plain SMILES under ``mol`` are embedded first."""
    payload = dict(payload)
    payload.setdefault("mol", payload["SMILES"])
    payload["mol"] = [
        Chem.RemoveHs(embed(mol)) if isinstance(mol, str) else mol
        for mol in payload["mol"]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def write_config(
    directory: Path, *, sources=(), corrections=None, **sections: dict
) -> Path:
    """Write a small pipeline.yaml (and corrections.yaml) and return its path.

    The stereo audit and the rebuild stage are off unless ``sections`` turns
    them on; every section given is merged over these defaults.
    """
    payload: dict[str, dict] = {
        "extraction": {"data_dir": "data", "sources": list(sources)},
        "validation": {
            "corrections_file": "corrections.yaml" if corrections else None,
            "audit_central_stereo": False,
        },
        "organometallic": {"enabled": False},
    }
    for section, values in sections.items():
        payload.setdefault(section, {}).update(values)
    if corrections:
        (directory / "corrections.yaml").write_text(
            yaml.safe_dump({"corrections": corrections})
        )
    path = directory / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    return PipelineConfig.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def build(config: PipelineConfig):
    """Every stage over the real data, built once per session."""
    return build_dataset(config)
