"""Tests for the pairwise enantiomer-ranking metric (chiral_docking headline)."""

from __future__ import annotations

import math

import numpy as np

from threedscriptors.data_handling.benchmarks import (
    BenchmarkManifest,
    EvalMetric,
    SplitVariant,
)
from threedscriptors.evaluation.benchmark.learners import LinearLearnerConfig
from threedscriptors.evaluation.benchmark.pairwise import pair_ranking_accuracy
from threedscriptors.evaluation.benchmark.runner import evaluate_pairwise_cell


def test_perfect_ranking_pools_conformers() -> None:
    # Two constitutions (mol 0 and 1), each with two enantiomers (iso a/b),
    # two conformers each. True winner is always the lower-scoring enantiomer.
    isomer_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    molecule_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # iso 0 (mol0) better than iso 1; iso 2 (mol1) better than iso 3.
    true = np.array([-5.0, -5.0, -4.0, -4.0, -7.0, -7.0, -6.0, -6.0])
    # Predictions preserve the order (with conformer noise that pooling cancels).
    pred = np.array([-4.9, -5.1, -4.1, -3.9, -7.2, -6.8, -6.1, -5.9])
    acc, n_pairs = pair_ranking_accuracy(true, pred, molecule_ids, isomer_ids)
    assert n_pairs == 2
    assert acc == 1.0


def test_reversed_ranking_scores_zero() -> None:
    isomer_ids = np.array([0, 1])
    molecule_ids = np.array([0, 0])
    true = np.array([-5.0, -4.0])  # iso 0 truly better
    pred = np.array([-3.0, -6.0])  # predicts iso 1 better -> wrong
    acc, n_pairs = pair_ranking_accuracy(true, pred, molecule_ids, isomer_ids)
    assert n_pairs == 1
    assert acc == 0.0


def test_tie_scores_half() -> None:
    # A chirality-invariant descriptor predicts identical scores for the pair.
    isomer_ids = np.array([0, 1])
    molecule_ids = np.array([0, 0])
    true = np.array([-5.0, -4.0])
    pred = np.array([-4.5, -4.5])
    acc, n_pairs = pair_ranking_accuracy(true, pred, molecule_ids, isomer_ids)
    assert n_pairs == 1
    assert acc == 0.5


def test_unpaired_constitution_is_skipped() -> None:
    # mol 1 has a single stereoisomer -> not a pair -> not scored.
    isomer_ids = np.array([0, 1, 2])
    molecule_ids = np.array([0, 0, 1])
    true = np.array([-5.0, -4.0, -3.0])
    pred = np.array([-5.0, -4.0, -3.0])
    acc, n_pairs = pair_ranking_accuracy(true, pred, molecule_ids, isomer_ids)
    assert n_pairs == 1
    assert acc == 1.0


def test_no_scorable_pairs_returns_nan() -> None:
    isomer_ids = np.array([0])
    molecule_ids = np.array([0])
    acc, n_pairs = pair_ranking_accuracy(
        np.array([-5.0]), np.array([-5.0]), molecule_ids, isomer_ids
    )
    assert n_pairs == 0
    assert math.isnan(acc)


def test_evaluate_pairwise_cell_end_to_end() -> None:
    """The runner cell fits a real Ridge probe then ranks the test pairs.

    Four constitutions (two train, two test), two enantiomers each, two
    conformers per enantiomer. The lone descriptor column encodes the true
    score (+ tiny conformer noise), so Ridge recovers the ordering and every
    test pair ranks correctly -> accuracy 1.0 over 2 pairs.
    """
    # 16 conformer rows: constitution c in 0..3, enantiomer e in {0,1}, conf k.
    mol_ids, iso_ids, y, X_col = [], [], [], []
    iso = 0
    for c in range(4):
        for e in range(2):
            # Distinct true scores so each pair has a clear winner.
            score = -5.0 - c - 0.5 * e
            for _k in range(2):
                mol_ids.append(c)
                iso_ids.append(iso)
                y.append(score)
                X_col.append(score + 0.01 * (_k - 0.5))
            iso += 1
    mol_ids = np.array(mol_ids)
    iso_ids = np.array(iso_ids)
    Y = np.array(y, dtype=float).reshape(-1, 1)
    X = np.array(X_col, dtype=float).reshape(-1, 1)

    # Constitutions 0,1 -> train; 2,3 -> test; no val rows needed by Ridge.
    tr = np.isin(mol_ids, [0, 1])
    te = np.isin(mol_ids, [2, 3])
    va = np.zeros_like(tr)
    splits = (tr, va, te)

    manifest = BenchmarkManifest(
        dataset_id="chiral_docking",
        metric=EvalMetric.pair_ranking_accuracy,
        split_variant=SplitVariant.predefined,
        source="local_chiro",
    )
    rows = evaluate_pairwise_cell(
        LinearLearnerConfig(ridge_alpha=0.01),
        splits,
        X,
        Y,
        mol_ids,
        iso_ids,
        "docking_top_score",
        manifest,
        "score_oracle",
        seed=0,
    )
    (row,) = rows
    assert row.metric_name == "pair-ranking-accuracy"
    assert row.target_col == "docking_top_score"
    assert row.n_test == 2  # two enantiomer pairs scored
    assert row.metric_value == 1.0
