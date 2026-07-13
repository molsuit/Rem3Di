from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remedi.configuration.dataset_analysis_config import (
    BitBirchConfig,
    BitBirchUmapConfig,
)
from remedi.configuration.dataset_comparison_config import (
    DatasetComparisonConfig,
    DatasetEntry,
    NnTanimotoConfig,
    ScaffoldOverlapConfig,
)
from remedi.configuration.dataset_config import DatasetConfig
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_comparison import (
    DatasetComparison,
    DatasetComparisonSummary,
)
from remedi.data_handling.dataset_creation.shard_aligned_writer import (
    ShardAlignedWriter,
)


def _build_small_dataset(
    tmp_path: Path,
    smiles_list: list[str],
    structures_per_mol: int = 1,
) -> Path:
    """Build a minimal on-disk MoleculeDataset and return its root path."""
    cfg = DatasetConfig(atom_chunk=64, molecule_chunk=16, contains_smiles=True)
    dataset = MoleculeDataset.create_empty_dataset(path=tmp_path, config=cfg)
    writer = ShardAlignedWriter(dataset)

    species = np.array([6, 8, 7, 1, 1], dtype=np.uint8)
    positions = np.tile(np.arange(15, dtype=np.float32).reshape(5, 3), (1, 1))

    n_mols = len(smiles_list)
    n_structures = n_mols * structures_per_mol
    all_positions = np.tile(positions, (n_structures, 1))
    all_species = np.tile(species, n_structures)
    ptr_cumsum = np.arange(1, n_structures + 1, dtype=np.int64) * 5
    mol_ids = np.repeat(np.arange(n_mols, dtype=np.int64), structures_per_mol)
    iso_ids = np.zeros(n_structures, dtype=np.int64)
    charges = np.zeros(n_structures, dtype=np.float32)
    mults = np.ones(n_structures, dtype=np.float32)

    writer.append_batch(
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
    writer.finalize()

    if dataset.smiles is not None:
        dataset.smiles.append_lines(smiles_list)
    if dataset.isomeric_smiles is not None:
        dataset.isomeric_smiles.append_lines(smiles_list)

    # Close mmaps before returning -- the comparison reopens the path.
    if dataset.smiles is not None:
        dataset.smiles.close()
    if dataset.isomeric_smiles is not None:
        dataset.isomeric_smiles.close()
    return tmp_path


# Two clearly distinct chemotypes per dataset so BitBIRCH should find structure.
PRETRAIN_SMILES = [
    "CCO",
    "CCN",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CC(C)O",
    "CC(C)N",
    "CC(C)CC",
    "CCC(C)CC",
]

EVAL_SMILES = [
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "c1cnc2ccccc2c1",
    "Cc1ccncc1",
    "Cn1cnc2ccccc21",
    "Oc1ccc2ccccc2c1",
]


@pytest.fixture
def two_small_datasets(tmp_path: Path) -> tuple[Path, Path]:
    a = _build_small_dataset(tmp_path / "ds_pretrain", PRETRAIN_SMILES)
    b = _build_small_dataset(tmp_path / "ds_eval", EVAL_SMILES)
    return a, b


def _make_config(
    pretrain_path: Path, eval_path: Path, out: Path
) -> DatasetComparisonConfig:
    return DatasetComparisonConfig(
        output_dir=out,
        random_seed=7,
        rdkit_n_workers=1,
        top_clusters_plotted=5,
        datasets=[
            DatasetEntry(name="pretrain_aliph", role="pretrain", path=pretrain_path),
            DatasetEntry(name="eval_arom", role="eval", path=eval_path),
        ],
        bitbirch=BitBirchConfig(
            enabled=True,
            fingerprint_kind="ecfp4",
            n_features=1024,
            threshold=0.65,
            branching_factor=50,
            merge_criterion="diameter",
            umap=BitBirchUmapConfig(enabled=False),
        ),
        nn_tanimoto=NnTanimotoConfig(
            enabled=True,
            fingerprint_kind="ecfp4",
            n_features=1024,
            chunk_size=8,
            backend="cpu",
            histogram_bins=20,
        ),
        scaffold_overlap=ScaffoldOverlapConfig(
            enabled=True,
            include_generic=True,
            n_rdkit_workers=1,
        ),
    )


def test_dataset_comparison_runs_end_to_end(
    two_small_datasets: tuple[Path, Path], tmp_path: Path
):
    pretrain_path, eval_path = two_small_datasets
    out = tmp_path / "comparison_out"
    config = _make_config(pretrain_path, eval_path, out)
    cmp = DatasetComparison(config)
    summary = cmp.run()
    cmp.output()

    assert isinstance(summary, DatasetComparisonSummary)
    # Per-dataset accounting
    assert {d.name for d in summary.datasets} == {"pretrain_aliph", "eval_arom"}
    name_to = {d.name: d for d in summary.datasets}
    assert name_to["pretrain_aliph"].n_fingerprints == len(PRETRAIN_SMILES)
    assert name_to["eval_arom"].n_fingerprints == len(EVAL_SMILES)

    # BitBIRCH ran and produced clusters
    assert summary.bitbirch.enabled
    assert summary.bitbirch.n_clusters >= 1
    assert summary.composition is not None
    assert summary.composition.n_clusters == summary.bitbirch.n_clusters
    # The two chemotypes are very different -- expect MI > 0.
    assert summary.composition.mutual_info > 0.0
    assert 0.0 <= summary.composition.normalized_mutual_info <= 1.0
    # Scaffold purity for non-empty clusters is in [0, 1]
    purity = summary.composition.scaffold_purity_distribution
    if purity.n > 0 and purity.min is not None and purity.max is not None:
        assert 0.0 <= purity.min <= purity.max <= 1.0

    # NN-Tanimoto ran for the one eval dataset
    assert summary.nn_tanimoto is not None
    assert summary.nn_tanimoto.n_pretrain_reference == len(PRETRAIN_SMILES)
    nn_per = {p.name: p for p in summary.nn_tanimoto.per_dataset}
    assert "eval_arom" in nn_per
    nn = nn_per["eval_arom"]
    assert nn.n_queries == len(EVAL_SMILES)
    assert 0.0 <= (nn.distribution.min or 0.0) <= 1.0
    assert 0.0 <= (nn.distribution.max or 0.0) <= 1.0
    # Aliphatic vs aromatic -> NN max similarities should be modest, not 1.0.
    assert (nn.distribution.max or 0.0) < 0.95

    # Scaffold overlap
    assert summary.scaffold_overlap is not None
    pair_key = "pretrain_aliph__eval_arom"
    # Aliphatic SMILES have empty Murcko scaffolds; aromatic ones don't ->
    # specific Jaccard should be ~0.
    assert summary.scaffold_overlap.pairwise_jaccard_specific.get(pair_key, 0.0) < 0.1
    assert "eval_arom" in summary.scaffold_overlap.eval_specific_coverage_by_pretrain

    # Output artifacts
    assert (out / "dataset_comparison_summary.yaml").exists()
    assert (out / "bitbirch_union_cluster_sizes.png").exists()
    assert (out / "cluster_composition.png").exists()
    assert (out / "cluster_scaffold_purity.png").exists()
    assert (out / "nn_tanimoto_distributions.png").exists()
    assert (out / "scaffold_jaccard_specific.png").exists()
    assert (out / "cluster_contingency.npz").exists()


def test_dataset_comparison_no_pretrain_skips_nn(
    two_small_datasets: tuple[Path, Path], tmp_path: Path
):
    """Without a pretrain entry the NN section degrades gracefully."""
    a, b = two_small_datasets
    out = tmp_path / "comparison_no_pretrain"
    cfg = _make_config(a, b, out)
    # Re-tag both entries as eval.
    cfg.datasets[0] = cfg.datasets[0].model_copy(update={"role": "eval"})
    cmp = DatasetComparison(cfg)
    summary = cmp.run()
    cmp.output()
    assert summary.nn_tanimoto is not None
    assert summary.nn_tanimoto.n_pretrain_reference == 0
    # Composition still produced, but pretrain stats are zero.
    assert summary.composition is not None
    assert summary.composition.pretrain_molecules_in_eval_clusters == 0
