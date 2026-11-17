"""Two-level Butina clustering of a dataset's molecules (GPU) and its cluster file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from remedi_data.chemistry.fingerprints import (
    morgan_fingerprints,
    tanimoto_similarity,
)
from remedi_data.selection import (
    HierarchicalClusters,
    SelectionConfig,
    read_selection,
    select_structures,
)
from remedi_data.yaml_io import read_model
from remedi_eval.dataset_analysis.molecule_clustering import (
    MoleculeClusteringConfig,
    cluster_molecules,
    cluster_size_summary,
    two_level_butina,
)
from testing import write_small_dataset

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Butina clustering runs on the GPU"
)

FAMILIES = [
    ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC"],
    ["CCCCCCCCO", "CCCCCCCCN", "CCCCCCCCC", "CCCCCCCCCC"],
    ["C1CCC2CCCCC2C1", "C1CCC2CCCCC2C1O"],
]


def test_the_fine_cutoff_must_be_below_the_coarse_one() -> None:
    with pytest.raises(ValueError, match="fine_cutoff"):
        MoleculeClusteringConfig(
            dataset=Path("x"), name="x", coarse_cutoff=0.4, fine_cutoff=0.6
        )


def test_cluster_size_summary() -> None:
    summary = cluster_size_summary(np.array([0, 0, 0, 1, 2]), "coarse")
    assert summary["coarse_clusters"] == 3
    assert summary["coarse_singletons"] == 2
    assert summary["coarse_largest"] == 3


@needs_gpu
def test_two_level_butina_nests_and_respects_the_cutoffs() -> None:
    smiles = [text for family in FAMILIES for text in family]
    fingerprints = morgan_fingerprints(smiles, layout="packed_words")
    clusters = two_level_butina(fingerprints, coarse_cutoff=0.8, fine_cutoff=0.4)
    for fine_id in np.unique(clusters.fine):
        assert len(np.unique(clusters.coarse[clusters.fine == fine_id])) == 1
    # Molecules at distance > coarse cutoff from everything else stay apart.
    for left in range(len(smiles)):
        for right in range(len(smiles)):
            distance = 1 - tanimoto_similarity(fingerprints[left], fingerprints[right])
            if clusters.coarse[left] == clusters.coarse[right]:
                assert distance <= 2 * 0.8 + 1e-9
    assert len(np.unique(clusters.fine)) >= len(np.unique(clusters.coarse))


@needs_gpu
def test_cluster_molecules_writes_a_file_selections_can_draw_from(
    tmp_path: Path,
) -> None:
    smiles = [text for family in FAMILIES for text in family]
    dataset = write_small_dataset(
        tmp_path,
        dataset_id="families",
        labels=[],
        metrics=None,
        targets=None,
        smiles=smiles,
    )
    record = cluster_molecules(
        MoleculeClusteringConfig(
            dataset=dataset, name="ecfp4", coarse_cutoff=0.8, fine_cutoff=0.4
        )
    )
    assert record.molecules == len(smiles)
    assignments = pl.read_parquet(dataset / "clusters/ecfp4.parquet")
    assert sorted(assignments["molecule_id"].to_list()) == list(range(len(smiles)))
    select_structures(
        dataset,
        SelectionConfig(
            name="diverse",
            steps=[
                HierarchicalClusters(
                    clusters="ecfp4",
                    count=record.coarse_clusters,
                    cluster_weighting="uniform",
                )
            ],
        ),
    )
    table = pl.read_parquet(dataset / "table.parquet")
    chosen = table.filter(
        pl.col("structure_id").is_in(read_selection(dataset, "diverse").tolist())
    )["molecule_id"]
    coarse_of = dict(
        zip(assignments["molecule_id"], assignments["coarse_cluster"], strict=True)
    )
    assert len({coarse_of[int(i)] for i in chosen}) == record.coarse_clusters
    again = cluster_molecules(
        MoleculeClusteringConfig(
            dataset=dataset, name="ecfp4", coarse_cutoff=0.8, fine_cutoff=0.4
        )
    )
    assert again.created_at == record.created_at


def test_the_production_clustering_config_parses() -> None:

    from remedi_eval.dataset_analysis.molecule_clustering import (
        MoleculeClusteringBatchConfig,
    )

    config = read_model(
        Path(__file__).resolve().parents[3]
        / "configs/dataset_creation/cluster_molecules.yaml",
        MoleculeClusteringBatchConfig,
    )
    assert {clustering.name for clustering in config.clusterings} == {"ecfp4_butina"}
