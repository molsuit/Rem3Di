from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remedi_data.runs.results import ArrayResult
from remedi_eval.dataset_analysis.comparison import (
    DatasetComparison,
    DatasetComparisonSummary,
    UnionFingerprints,
)
from remedi_eval.dataset_analysis.config import (
    BitBirchConfig,
    BitBirchUmapConfig,
    CpuNnTanimotoBackend,
    DatasetComparisonConfig,
    DatasetEntry,
    NnTanimotoConfig,
    ScaffoldOverlapConfig,
)
from testing import write_synthetic_dataset


def _written(directory: Path, smiles: list[str]) -> Path:
    """Write a dataset; the comparison reopens the path itself."""
    write_synthetic_dataset(directory, smiles)
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
    a = _written(tmp_path / "ds_pretrain", PRETRAIN_SMILES)
    b = _written(tmp_path / "ds_eval", EVAL_SMILES)
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
            radius=2,
            fingerprint_size=1024,
            threshold=0.65,
            branching_factor=50,
            merge_criterion="diameter",
            umap=BitBirchUmapConfig(enabled=False),
        ),
        nn_tanimoto=NnTanimotoConfig(enabled=True, backend=CpuNnTanimotoBackend()),
        scaffold_overlap=ScaffoldOverlapConfig(enabled=True, include_generic=True),
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
    assert name_to["pretrain_aliph"].n_smiles_unique == len(PRETRAIN_SMILES)
    assert name_to["eval_arom"].n_smiles_unique == len(EVAL_SMILES)

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


TRANS_DECALIN = "C1CC[C@H]2CCCC[C@@H]2C1"
CIS_DECALIN = "C1CC[C@H]2CCCC[C@H]2C1"


def test_scaffold_purity_skips_acyclic_molecules_and_keeps_chirality(
    tmp_path: Path,
) -> None:
    """Purity uses the splits' scaffold definition: chirality kept, and the
    empty scaffold of an acyclic molecule is no scaffold at all."""
    config = _make_config(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    comparison = DatasetComparison(config)
    union = UnionFingerprints(
        fps_packed=np.zeros((6, 128), dtype=np.uint8),
        dataset_index=np.array([0, 0, 1, 1, 0, 1]),
        smiles=["CCO", "CCCN", TRANS_DECALIN, CIS_DECALIN, "c1ccccc1", "Cc1ccccc1"],
    )
    # Cluster 0: acyclic only; 1: the two decalin diastereomers; 2: benzene ring.
    row_cluster = np.array([0, 0, 1, 1, 2, 2])

    composition = comparison.analyze_clusters(union, row_cluster)

    assert composition is not None
    purity = composition.scaffold_purity_distribution
    assert (purity.n, purity.min, purity.max) == (2, 0.5, 1.0)

    out = comparison.output(tmp_path / "out")
    # Plain string arrays: the npz loads without allow_pickle.
    arrays = ArrayResult.load(out / "cluster_contingency.npz")
    assert list(arrays["dataset_names"]) == ["pretrain_aliph", "eval_arom"]


def test_disabled_bitbirch_is_not_run(
    two_small_datasets: tuple[Path, Path], tmp_path: Path
) -> None:
    pretrain_path, eval_path = two_small_datasets
    config = _make_config(pretrain_path, eval_path, tmp_path / "out")
    config.bitbirch.enabled = False

    summary = DatasetComparison(config).run()

    assert summary.bitbirch.enabled is False
    assert summary.composition is None
    assert summary.nn_tanimoto is not None
    assert summary.nn_tanimoto.per_dataset


def test_bitbirch_max_molecules_caps_the_union(
    two_small_datasets: tuple[Path, Path], tmp_path: Path
) -> None:
    pretrain_path, eval_path = two_small_datasets
    config = _make_config(pretrain_path, eval_path, tmp_path / "out")
    config.bitbirch.max_molecules = 8

    summary = DatasetComparison(config).run()

    assert summary.bitbirch.n_fingerprints == 8
    assert summary.composition is not None
    assert (
        summary.composition.cluster_size_distribution.n == summary.bitbirch.n_clusters
    )


def test_nvmolkit_backend_needs_whole_fingerprint_words(tmp_path: Path) -> None:
    from remedi_eval.dataset_analysis.config import NvmolkitNnTanimotoBackend

    config = _make_config(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    with pytest.raises(ValueError, match="multiple of 32"):
        DatasetComparisonConfig.model_validate(
            config.model_dump()
            | {
                "bitbirch": config.bitbirch.model_dump() | {"fingerprint_size": 1000},
                "nn_tanimoto": {"backend": NvmolkitNnTanimotoBackend().model_dump()},
            }
        )


def test_each_smiles_gets_its_scaffold_once(
    two_small_datasets: tuple[Path, Path], tmp_path: Path, monkeypatch
) -> None:
    """Cluster purity and the scaffold overlap share one scaffold per SMILES."""
    import remedi_eval.dataset_analysis.comparison as comparison_module

    computed: list[tuple[str, bool]] = []
    original = comparison_module._scaffold_of

    def counting(smiles: str, generic: bool) -> str | None:
        computed.append((smiles, generic))
        return original(smiles, generic)

    monkeypatch.setattr(comparison_module, "_scaffold_of", counting)
    pretrain_path, eval_path = two_small_datasets
    DatasetComparison(_make_config(pretrain_path, eval_path, tmp_path / "out")).run()
    assert len(computed) == len(set(computed))
    assert {smiles for smiles, generic in computed if not generic} == set(
        PRETRAIN_SMILES + EVAL_SMILES
    )
