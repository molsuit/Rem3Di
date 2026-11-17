"""The shared BitBIRCH helpers: cluster bookkeeping and the UMAP sample."""

from __future__ import annotations

import numpy as np
import pytest

from remedi_eval.dataset_analysis.bitbirch import (
    BitBirchClusters,
    fit_bitbirch,
    umap_sample_rows,
)
from remedi_eval.dataset_analysis.config import BitBirchConfig, BitBirchUmapConfig


def _clusters() -> BitBirchClusters:
    # Cluster 0 holds 80 rows, clusters 1..20 one row each.
    row_cluster = np.concatenate([np.zeros(80, dtype=np.int64), np.arange(1, 21)])
    return BitBirchClusters(
        row_cluster=row_cluster,
        sizes=np.bincount(row_cluster),
        threshold=0.5,
        threshold_source="config",
        mean_isim=None,
        std_isim=None,
    )


def test_uniform_sampling_draws_from_every_row() -> None:
    config = BitBirchUmapConfig(sample_size=10, sampling="uniform")
    rows = umap_sample_rows(_clusters(), config, np.random.default_rng(0))
    assert rows.size == 10 and np.all(np.diff(rows) > 0)
    whole = umap_sample_rows(
        _clusters(), BitBirchUmapConfig(sample_size=None), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(whole, np.arange(100))


def test_largest_clusters_first_keeps_the_largest_clusters_visible() -> None:
    """The largest clusters' members take at least half the budget although
    the small clusters outnumber them."""
    config = BitBirchUmapConfig(
        sample_size=10, sampling="largest_clusters_first", top_clusters_colored=1
    )
    rows = umap_sample_rows(_clusters(), config, np.random.default_rng(0))
    assert rows.size == 10
    assert int((_clusters().row_cluster[rows] == 0).sum()) >= 5


def test_largest_orders_the_clusters_by_size() -> None:
    assert _clusters().largest(2)[0] == 0
    assert _clusters().largest(0).size == 0


def test_fit_bitbirch_assigns_every_row() -> None:
    pytest.importorskip("bblean")
    from remedi_data.chemistry.fingerprints import morgan_fingerprints

    smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1"]
    fingerprints = morgan_fingerprints(smiles, layout="packed_bytes", size=256)
    clusters = fit_bitbirch(
        fingerprints, BitBirchConfig(fingerprint_size=256, threshold=0.3)
    )
    assert clusters.row_cluster.min() >= 0
    assert int(clusters.sizes.sum()) == len(smiles)
    summary = clusters.summary(len(smiles), None)
    assert summary.n_fingerprints == len(smiles)
    assert summary.n_clusters == clusters.sizes.size
