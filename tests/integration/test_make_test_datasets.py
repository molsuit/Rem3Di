"""The stand-in corpora of scripts/gpu_checks/make_test_datasets.py: molecule lists
of the sizes the GPU check scripts select, and a written dataset with its
selection that the pretraining mixture resolves."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rdkit import Chem

from remedi.training.data.mixture import (
    DatasetComponent,
    MoleculeFractionValidation,
    PretrainingDataConfig,
    resolve_mixture,
)

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/gpu_checks/make_test_datasets.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_test_datasets = _load_script()


def test_small_molecules_are_distinct_and_pcqm_sized() -> None:
    smiles = make_test_datasets.small_molecules(300)
    assert len(smiles) == 300
    assert len(set(smiles)) == 300
    smallest, largest = make_test_datasets.SMALL_HEAVY_ATOMS
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        assert molecule is not None
        assert smallest <= molecule.GetNumHeavyAtoms() <= largest


def test_large_molecules_are_in_the_benchmark_atom_range() -> None:
    smiles = make_test_datasets.large_molecules()
    assert len(smiles) >= 5
    smallest, largest = make_test_datasets.LARGE_ATOMS_WITH_HYDROGENS
    for value in smiles:
        atoms = make_test_datasets.atoms_with_hydrogens(value)
        assert smallest <= atoms <= largest


def test_a_written_corpus_resolves_in_a_mixture(tmp_path: Path) -> None:
    smiles = make_test_datasets.small_molecules(12)
    make_test_datasets.write_corpus(tmp_path, "pcqm4mv2", smiles, "diverse_100k")
    resolved = resolve_mixture(
        PretrainingDataConfig(
            domain="drug_like",
            components=[
                DatasetComponent(
                    dataset=tmp_path / "pcqm4mv2", selection="diverse_100k"
                )
            ],
            validation=MoleculeFractionValidation(fraction=0.25, seed=0),
        )
    )
    part = resolved.components[0].parts[0]
    assert len(part.train_structure_ids) + len(part.validation_structure_ids) == 12
