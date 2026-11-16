"""Tests for the pairwise enantiomer-ranking metric (chiral_docking headline).

Pairs come from the dataset table's ``enantiomer_of`` column (invariant 4), not
from "two stereoisomers of one constitution": a diastereomer pair is not an
enantiomer pair and must not be scored.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from remedi_data.bundle import EvalMetric
from remedi_eval.benchmark.cells import BenchmarkCell, evaluate_pairwise_cell
from remedi_eval.benchmark.learners import LinearLearner
from remedi_eval.benchmark.pairwise import pair_ranking_accuracy
from remedi_eval.protocol import Partition

MISSING = pd.NA


def _partners(*values) -> pd.api.extensions.ExtensionArray:
    return pd.array(list(values), dtype="Int64")


def test_perfect_ranking_pools_conformers() -> None:
    # Two enantiomer pairs (0 <-> 1, 2 <-> 3), two conformers each. The true
    # winner is always the lower-scoring enantiomer.
    stereoisomer_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    enantiomer_of = _partners(1, 1, 0, 0, 3, 3, 2, 2)
    true = np.array([-5.0, -5.0, -4.0, -4.0, -7.0, -7.0, -6.0, -6.0])
    # Predictions preserve the order (with conformer noise that pooling cancels).
    predicted = np.array([-4.9, -5.1, -4.1, -3.9, -7.2, -6.8, -6.1, -5.9])
    accuracy, n_pairs = pair_ranking_accuracy(
        true, predicted, stereoisomer_ids, enantiomer_of
    )
    assert n_pairs == 2
    assert accuracy == 1.0


def test_reversed_ranking_scores_zero() -> None:
    accuracy, n_pairs = pair_ranking_accuracy(
        np.array([-5.0, -4.0]),  # stereoisomer 0 truly better
        np.array([-3.0, -6.0]),  # predicts stereoisomer 1 better -> wrong
        np.array([0, 1]),
        _partners(1, 0),
    )
    assert (accuracy, n_pairs) == (0.0, 1)


def test_tie_scores_half() -> None:
    # A chirality-invariant descriptor predicts identical scores for the pair.
    accuracy, n_pairs = pair_ranking_accuracy(
        np.array([-5.0, -4.0]),
        np.array([-4.5, -4.5]),
        np.array([0, 1]),
        _partners(1, 0),
    )
    assert (accuracy, n_pairs) == (0.5, 1)


def test_diastereomers_are_not_scored() -> None:
    """Two stereoisomers of one constitution that are not mirror images (no
    ``enantiomer_of``) are a diastereomer pair, not an enantiomer pair."""
    accuracy, n_pairs = pair_ranking_accuracy(
        np.array([-5.0, -4.0]),
        np.array([-5.0, -4.0]),
        np.array([0, 1]),
        _partners(MISSING, MISSING),
    )
    assert n_pairs == 0
    assert math.isnan(accuracy)


def test_two_enantiomer_pairs_of_one_constitution_are_both_scored() -> None:
    """RR/SS and RS/SR of a two-centre molecule: four stereoisomers of one
    constitution, two enantiomer pairs, each ranked on its own."""
    stereoisomer_ids = np.array([10, 11, 12, 13])
    enantiomer_of = _partners(11, 10, 13, 12)
    true = np.array([-5.0, -4.0, -6.0, -7.0])
    predicted = np.array([-5.0, -4.0, -6.5, -6.0])  # first right, second wrong
    accuracy, n_pairs = pair_ranking_accuracy(
        true, predicted, stereoisomer_ids, enantiomer_of
    )
    assert (accuracy, n_pairs) == (0.5, 2)


def test_a_pair_split_across_folds_is_skipped() -> None:
    # Stereoisomer 0's partner (1) is not in the scored fold.
    accuracy, n_pairs = pair_ranking_accuracy(
        np.array([-5.0, -3.0, -2.0]),
        np.array([-5.0, -3.0, -2.0]),
        np.array([0, 2, 3]),
        _partners(1, 3, 2),
    )
    assert (accuracy, n_pairs) == (1.0, 1)


def test_evaluate_pairwise_cell_end_to_end() -> None:
    """The runner cell fits a real Ridge probe then ranks the test pairs.

    Four constitutions (two train, two test), two enantiomers each, two
    conformers per enantiomer. The lone descriptor column encodes the true
    score (+ tiny conformer noise), so Ridge recovers the ordering and every
    test pair ranks correctly -> accuracy 1.0 over 2 pairs.
    """
    constitution, stereoisomer_ids, partners, y, feature = [], [], [], [], []
    for constitution_index in range(4):
        for mirror in range(2):
            stereoisomer = 2 * constitution_index + mirror
            # Distinct true scores so each pair has a clear winner.
            score = -5.0 - constitution_index - 0.5 * mirror
            for conformer in range(2):
                constitution.append(constitution_index)
                stereoisomer_ids.append(stereoisomer)
                partners.append(stereoisomer ^ 1)
                y.append(score)
                feature.append(score + 0.01 * (conformer - 0.5))
    constitution = np.array(constitution)
    stereoisomers = pd.DataFrame(
        {
            "stereoisomer_id": np.array(stereoisomer_ids, dtype=np.int64),
            "enantiomer_of": pd.array(partners, dtype="Int64"),
        }
    )
    Y = np.array(y, dtype=float).reshape(-1, 1)
    X = np.array(feature, dtype=float).reshape(-1, 1)

    # Constitutions 0,1 -> train; 2,3 -> test; no val rows needed by Ridge.
    partition = Partition(
        split_column="split",
        train=np.flatnonzero(np.isin(constitution, [0, 1])),
        valid=np.array([], dtype=np.int64),
        test=np.flatnonzero(np.isin(constitution, [2, 3])),
    )
    cell = BenchmarkCell(
        dataset_id="chiral_docking",
        metric=EvalMetric.pair_ranking_accuracy,
        split_column="split",
        seed=0,
    )
    rows = evaluate_pairwise_cell(
        LinearLearner(ridge_alpha=0.01),
        partition,
        X,
        Y,
        stereoisomers,
        "docking_top_score",
        cell,
        "score_oracle",
    )
    (row,) = rows
    assert row.metric_name == "pair-ranking-accuracy"
    assert row.target_col == "docking_top_score"
    assert row.n_test == 2  # two enantiomer pairs scored
    assert row.metric_value == 1.0
    assert (row.dataset_id, row.split_column, row.seed) == (
        "chiral_docking",
        "split",
        0,
    )
