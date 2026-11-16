"""Pairwise enantiomer-ranking metric for the Chiro docking benchmark.

The descriptor probe regresses a per-conformer docking ``top_score``. Scoring is
*relative*, not absolute: for each enantiomer pair the two mirror images are
ranked by their predicted score and the metric is the fraction of pairs whose
predicted better-docker (lower score) matches the truth. The pairs come from the
dataset table's ``enantiomer_of`` column (invariant 4: symmetric, irreflexive,
within one ``molecule_id``), so two diastereomers of one constitution are never
scored as a pair. That needs the per-structure ``stereoisomer_id`` and
``enantiomer_of``, so the metric lives here rather than in the uniform
``(y_true, y_pred)`` registry in :mod:`metrics`.

A chirality-*invariant* descriptor returns identical features for an enantiomer
pair (same graph, mirror-image geometries share an interatomic-distance matrix),
hence identical predicted scores -> ties, which this scores as 0.5. So the metric
reads ~0.5 for a chirality-blind descriptor and only climbs above chance when the
representation actually resolves the stereocentre -- exactly the diagnostic the
benchmark is built to expose.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def pair_ranking_accuracy(
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
    stereoisomer_ids: np.ndarray,
    enantiomer_of: Sequence[object] | pd.api.extensions.ExtensionArray | np.ndarray,
) -> tuple[float, int]:
    """Fraction of enantiomer pairs ranked correctly by predicted ``top_score``.

    All four arrays are per-conformer and aligned (the scored fold).
    ``enantiomer_of`` is the ``stereoisomer_id`` of each row's mirror image,
    missing (``pd.NA`` / NaN) when it has none. Conformer scores are pooled
    (mean) per stereoisomer; a pair is scored when both mirror images are in
    the fold and their true pooled scores differ. A pair counts 1.0 when the
    predicted lower score lands on the truly-lower-scoring enantiomer, 0.0 when
    reversed, and 0.5 on a predicted tie. Returns ``(accuracy, n_pairs)``;
    accuracy is NaN when no scorable pair exists.
    """
    per_stereoisomer = (
        pd.DataFrame(
            {
                "stereoisomer_id": np.asarray(stereoisomer_ids, dtype=np.int64),
                "enantiomer_of": pd.array(enantiomer_of, dtype="Int64"),
                "true": np.asarray(true_scores, dtype=float),
                "predicted": np.asarray(predicted_scores, dtype=float),
            }
        )
        .groupby("stereoisomer_id")
        .agg(
            enantiomer_of=("enantiomer_of", "first"),
            true=("true", "mean"),
            predicted=("predicted", "mean"),
        )
    )
    partner = per_stereoisomer["enantiomer_of"]
    # Each pair once (from its lower id), and only when the partner is in the fold.
    first = per_stereoisomer[
        partner.notna()
        & (per_stereoisomer.index < partner)
        & partner.isin(per_stereoisomer.index)
    ]
    second = per_stereoisomer.loc[first["enantiomer_of"].astype(np.int64)]
    true_difference = first["true"].to_numpy() - second["true"].to_numpy()
    predicted_difference = (
        first["predicted"].to_numpy() - second["predicted"].to_numpy()
    )

    # No ground-truth winner (margin filtering should prevent it): not scored.
    decided = true_difference != 0.0
    true_difference = true_difference[decided]
    predicted_difference = predicted_difference[decided]
    if true_difference.size == 0:
        return float("nan"), 0
    scores = np.where(
        predicted_difference == 0.0,
        0.5,
        (np.sign(predicted_difference) == np.sign(true_difference)).astype(float),
    )
    return float(scores.mean()), int(scores.size)


__all__ = ["pair_ranking_accuracy"]
