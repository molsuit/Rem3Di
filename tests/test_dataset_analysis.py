from pathlib import Path

import numpy as np
import pytest

from threedscriptors.configuration.dataset_analysis_config import (
    BitBirchConfig,
    MoleculeDatasetAnalysisConfig,
)
from threedscriptors.configuration.dataset_config import DatasetConfig
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import (
    DatasetSummary,
    MoleculeDatasetAnalysis,
    _compute_descriptors,
)


def _build_small_dataset(
    tmp_path: Path,
    smiles_list: list[str],
    structures_per_mol: int = 2,
) -> MoleculeDataset:
    """Build a minimal on-disk dataset with `structures_per_mol` copies per SMILES."""
    cfg = DatasetConfig(atom_chunk=64, molecule_chunk=16, contains_smiles=True)
    dataset = MoleculeDataset.create_empty_dataset(path=tmp_path, config=cfg)

    # Fake atomic data: each structure has 5 atoms (e.g., C, O, N, H, H)
    species = np.array([6, 8, 7, 1, 1], dtype=np.uint8)
    positions = np.tile(np.arange(15, dtype=np.float32).reshape(5, 3), (1, 1))

    n_mols = len(smiles_list)
    n_structures = n_mols * structures_per_mol
    all_positions = np.tile(positions, (n_structures, 1))
    all_species = np.tile(species, n_structures)

    # cumulative ends per structure
    ptr_cumsum = np.arange(1, n_structures + 1, dtype=np.int64) * 5
    mol_ids = np.repeat(np.arange(n_mols, dtype=np.int64), structures_per_mol)
    iso_ids = np.zeros(n_structures, dtype=np.int64)
    charges = np.zeros(n_structures, dtype=np.float32)
    mults = np.ones(n_structures, dtype=np.float32)

    dataset.append_batch(
        positions=all_positions,
        atomic_numbers=all_species,
        batch_ptr_cumsum=ptr_cumsum,
        molecule_ids=mol_ids,
        stereoisomer_ids=iso_ids,
        total_charge=charges,
        multiplicity=mults,
        system_targets=None,
        system_masks=None,
        atom_targets=None,
        atom_masks=None,
    )
    dataset.shrink_to_fit()

    # Append the SMILES separately (one per unique molecule).
    if dataset.smiles is not None:
        dataset.smiles.append_lines(smiles_list)
    if dataset.isomeric_smiles is not None:
        dataset.isomeric_smiles.append_lines(smiles_list)

    return dataset


def test_compute_descriptors_valid_and_invalid():
    d = _compute_descriptors("CCO")
    assert d.valid is True
    assert d.mw is not None and d.mw > 0
    assert d.n_heavy_atoms == 3
    assert d.scaffold == ""  # ethanol has no ring scaffold

    bad = _compute_descriptors("not a smiles")
    assert bad.valid is False


def test_distribution_stats_handles_empty():
    from threedscriptors.data_handling.dataset_analysis import DistributionStats

    s = DistributionStats.from_array(np.asarray([], dtype=np.float64))
    assert s.n == 0
    assert s.mean is None


def test_molecule_dataset_analysis_runs(tmp_path: Path):
    smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "COc1ccc(CC(N)C(=O)O)cc1",
        "NC(=O)c1ccncc1",
        "C[C@H](N)C(=O)O",
    ]
    dataset = _build_small_dataset(tmp_path / "ds", smiles, structures_per_mol=3)

    config = MoleculeDatasetAnalysisConfig(
        rdkit_subsample=None,
        rdkit_n_workers=1,
        atom_chunk_size=32,
        structure_chunk_size=4,
        max_example_molecules=4,
        bitbirch=BitBirchConfig(enabled=False),
    )
    analysis = MoleculeDatasetAnalysis(dataset, config=config)
    summary = analysis.run()

    assert isinstance(summary, DatasetSummary)
    assert summary.n_structures == len(smiles) * 3
    assert summary.n_atoms == summary.n_structures * 5
    assert summary.atoms_per_structure.mean == 5.0
    assert summary.heteroatoms_per_structure.mean == 2.0  # O and N hetero atoms
    assert summary.atom_species_counts["H"] == summary.n_structures * 2
    assert 0.0 <= (summary.lipinski_ro5_pass_fraction or 0.0) <= 1.0
    assert summary.stereocentre_ratio is not None
    assert summary.n_unique_scaffolds is not None and summary.n_unique_scaffolds >= 1
    # Outputs include at least the summary yaml + figures
    out_dir = tmp_path / "out"
    analysis.output(out_dir)
    assert (out_dir / "dataset_summary.yaml").exists()
    assert (out_dir / "drug_likeness_distributions.png").exists()
    assert (out_dir / "molecule_size_distribution.png").exists()


def test_heteroatom_chunking_matches_naive(tmp_path: Path):
    smiles = ["CCO", "c1ccncc1", "CCN"]
    dataset = _build_small_dataset(tmp_path / "ds", smiles, structures_per_mol=2)

    config = MoleculeDatasetAnalysisConfig(
        rdkit_subsample=0,
        rdkit_n_workers=1,
        atom_chunk_size=3,
        structure_chunk_size=2,
        bitbirch=BitBirchConfig(enabled=False),
    )
    analysis = MoleculeDatasetAnalysis(dataset, config=config)
    chunked = analysis.heteroatom_counts_per_structure()

    # Naive reference: all 5-atom structures have O+N=2 heteroatoms.
    expected = np.full(dataset.N_structures, 2, dtype=np.int64)
    np.testing.assert_array_equal(chunked, expected)


def test_bitbirch_runs_when_available(tmp_path: Path):
    bblean = pytest.importorskip("bblean")  # noqa: F841

    smiles = [
        "CCO",
        "CCN",
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc2ccccc2c1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "COc1ccc(CC(N)C(=O)O)cc1",
        "NC(=O)c1ccncc1",
    ]
    dataset = _build_small_dataset(tmp_path / "ds", smiles, structures_per_mol=1)

    config = MoleculeDatasetAnalysisConfig(
        rdkit_subsample=None,
        rdkit_n_workers=1,
        max_example_molecules=2,
        bitbirch=BitBirchConfig(
            enabled=True,
            fingerprint_kind="ecfp4",
            n_features=512,
            threshold=0.65,
            branching_factor=10,
            merge_criterion="diameter",
            max_molecules=None,
        ),
    )
    analysis = MoleculeDatasetAnalysis(dataset, config=config)
    summary = analysis.run()
    assert summary.bitbirch is not None
    assert summary.bitbirch.enabled is True
    assert summary.bitbirch.n_input_smiles == len(smiles)
    assert summary.bitbirch.n_clusters >= 1
