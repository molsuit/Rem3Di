from pathlib import Path

import numpy as np
import pytest

from remedi_data.testing import write_synthetic_dataset
from remedi_eval.dataset_analysis.analysis import (
    DatasetSummary,
    MoleculeDatasetAnalysis,
    _compute_descriptors,
)
from remedi_eval.dataset_analysis.config import (
    BitBirchConfig,
    BitBirchUmapConfig,
    MoleculeDatasetAnalysisConfig,
)
from remedi_eval.projection import UmapConfig


def test_compute_descriptors_valid_and_invalid():
    d = _compute_descriptors("CCO")
    assert d.valid is True
    assert d.mw is not None and d.mw > 0
    assert d.n_heavy_atoms == 3
    assert d.scaffold == ""  # ethanol has no ring scaffold

    bad = _compute_descriptors("not a smiles")
    assert bad.valid is False


def test_compute_descriptors_scaffold_keeps_chirality_like_the_splits():
    """The analysis scaffold is the one the scaffold splits group by."""
    from remedi_data.chemistry.splits import murcko_scaffold

    trans_decalin = "C1CC[C@H]2CCCC[C@@H]2C1"
    scaffold = _compute_descriptors(trans_decalin).scaffold
    assert scaffold is not None and "@" in scaffold
    assert scaffold == murcko_scaffold(trans_decalin)


def test_distribution_stats_handles_empty():
    from remedi_eval.dataset_analysis.statistics import DistributionStats

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
    dataset = write_synthetic_dataset(
        tmp_path / "ds", smiles, structures_per_molecule=3
    )

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


def test_target_stats_come_from_the_spec_labels_in_the_table(tmp_path: Path):
    from remedi_data.bundle import LabelColumn, TaskType
    from remedi_data.dataset.molecule_dataset import MoleculeDataset
    from remedi_data.testing import write_small_dataset

    values = np.linspace(0.0, 9.0, 10)
    values[3] = np.nan
    directory = write_small_dataset(
        tmp_path,
        dataset_id="labelled",
        labels=[LabelColumn(name="gap", task_type=TaskType.regression)],
        metrics=None,
        targets=values.reshape(-1, 1),
    )
    analysis = MoleculeDatasetAnalysis(
        MoleculeDataset.open_existing_dataset_from_dir(directory)
    )

    (stats,) = analysis.target_stats()

    assert stats.name == "gap"
    assert stats.coverage == pytest.approx(0.9)
    assert stats.distribution.n == 9
    assert stats.distribution.mean == pytest.approx(np.nanmean(values))
    # A dataset without dataset.yaml (no declared labels) has no target stats.
    synthetic = write_synthetic_dataset(tmp_path / "synthetic", ["CCO"])
    assert MoleculeDatasetAnalysis(synthetic).target_stats() == []


def test_heteroatom_chunking_matches_naive(tmp_path: Path):
    smiles = ["CCO", "c1ccncc1", "CCN"]
    dataset = write_synthetic_dataset(
        tmp_path / "ds", smiles, structures_per_molecule=2
    )

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
    pytest.importorskip("bblean")

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
    dataset = write_synthetic_dataset(
        tmp_path / "ds", smiles, structures_per_molecule=1
    )

    config = MoleculeDatasetAnalysisConfig(
        rdkit_subsample=None,
        rdkit_n_workers=1,
        max_example_molecules=2,
        bitbirch=BitBirchConfig(
            enabled=True,
            radius=2,
            fingerprint_size=512,
            threshold=0.65,
            branching_factor=10,
            merge_criterion="diameter",
            max_molecules=None,
            umap=BitBirchUmapConfig(enabled=False),
        ),
    )
    analysis = MoleculeDatasetAnalysis(dataset, config=config)
    summary = analysis.run()
    assert summary.bitbirch is not None
    assert summary.bitbirch.enabled is True
    assert summary.bitbirch.n_input_smiles == len(smiles)
    assert summary.bitbirch.n_clusters >= 1
    assert summary.bitbirch.umap is not None and summary.bitbirch.umap.enabled is False


def test_bitbirch_umap_projection_runs(tmp_path: Path):
    pytest.importorskip("bblean")
    pytest.importorskip("umap")

    rng = np.random.default_rng(0)
    # Build a diverse-enough set so UMAP with low n_neighbors converges.
    seed_smiles = [
        "CCO",
        "CCN",
        "CCC",
        "CCCO",
        "CCCN",
        "CCCC",
        "CCCCO",
        "CCCCN",
        "c1ccccc1",
        "c1ccncc1",
        "c1ccoc1",
        "c1ccsc1",
        "Cc1ccccc1",
        "Nc1ccccc1",
        "Oc1ccccc1",
        "Clc1ccccc1",
        "C1CCCCC1",
        "C1CCNCC1",
        "C1CCOCC1",
        "C1CCSCC1",
    ]
    smiles = seed_smiles * 4  # 80 entries; UMAP handles this fine
    rng.shuffle(smiles)
    dataset = write_synthetic_dataset(
        tmp_path / "ds", smiles, structures_per_molecule=1
    )

    config = MoleculeDatasetAnalysisConfig(
        rdkit_subsample=None,
        rdkit_n_workers=1,
        max_example_molecules=2,
        bitbirch=BitBirchConfig(
            enabled=True,
            radius=2,
            fingerprint_size=512,
            threshold=0.65,
            branching_factor=10,
            merge_criterion="diameter",
            max_molecules=None,
            umap=BitBirchUmapConfig(
                enabled=True,
                sample_size=None,
                projection=UmapConfig(
                    n_neighbors=5, min_dist=0.1, metric="jaccard", random_state=0
                ),
                top_clusters_colored=5,
                plot_size=300,
            ),
        ),
    )
    analysis = MoleculeDatasetAnalysis(dataset, config=config)
    summary = analysis.run()
    assert summary.bitbirch is not None
    assert summary.bitbirch.umap is not None
    assert summary.bitbirch.umap.enabled is True
    assert summary.bitbirch.umap.n_points > 0
    projection = summary.bitbirch.umap.projection
    assert projection is not None and projection.metric == "jaccard"

    out_dir = tmp_path / "out"
    analysis.output(out_dir)
    assert (out_dir / "bitbirch_umap.png").exists()
    assert (out_dir / "bitbirch_umap_legend.png").exists()
    assert (out_dir / "bitbirch_umap_coords.npz").exists()


def test_datashader_image_result_returns_the_standard_manifest_entry(tmp_path: Path):
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    from remedi_eval.dataset_analysis.analysis import _DatashaderImageResult

    points = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    image = tf.shade(ds.Canvas(plot_width=8, plot_height=8).points(points, "x", "y"))
    result = _DatashaderImageResult(file_name=Path("umap.png"), image=image)

    entry = result.serialize_to(tmp_path)

    assert entry == {"result_type": "datashader_image", "file_name": "umap.png"}
    assert (tmp_path / "umap.png").exists()
