from __future__ import annotations

from pathlib import Path

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
from remedi.data_handling.dataset_comparison import (
    DatasetComparison,
    DatasetComparisonSummary,
)
from remedi_data.testing import write_dataset_with_smiles_stores


def _written_and_closed(directory: Path, smiles: list[str]) -> Path:
    """Write a dataset and close it; the comparison reopens the path itself."""
    write_dataset_with_smiles_stores(directory, smiles).close()
    return directory


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
    a = _written_and_closed(tmp_path / "ds_pretrain", PRETRAIN_SMILES)
    b = _written_and_closed(tmp_path / "ds_eval", EVAL_SMILES)
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
