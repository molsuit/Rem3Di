"""Pairwise enantiomer-ranking metric for the Chiro docking benchmark.

The descriptor probe regresses a per-conformer docking ``top_score``. Scoring is
*relative*, not absolute: for each constitution the two enantiomers are ranked by
their predicted score and the metric is the fraction of pairs whose predicted
better-docker (lower score) matches the truth. This needs the molecule (the
``molecule_id`` constitution group) and stereoisomer (``isomer_id`` enantiomer
group) ids, so it lives here rather than in the uniform ``(y_true, y_pred)``
metric registry in :mod:`metrics`.

A chirality-*invariant* descriptor returns identical features for an enantiomer
pair (same graph, mirror-image geometries share an interatomic-distance matrix),
hence identical predicted scores -> ties, which this scores as 0.5. So the metric
reads ~0.5 for a chirality-blind descriptor and only climbs above chance when the
representation actually resolves the stereocentre -- exactly the diagnostic the
benchmark is built to expose.
"""

from __future__ import annotations

import numpy as np


def _aggregate_per_isomer(
    values: np.ndarray, isomer_ids: np.ndarray
) -> dict[int, float]:
    """Mean of ``values`` within each ``isomer_id`` group (pool conformers)."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for v, iso in zip(values, isomer_ids, strict=True):
        iso = int(iso)
        sums[iso] = sums.get(iso, 0.0) + float(v)
        counts[iso] = counts.get(iso, 0) + 1
    return {iso: sums[iso] / counts[iso] for iso in sums}


def pair_ranking_accuracy(
    true_scores: np.ndarray,
    pred_scores: np.ndarray,
    molecule_ids: np.ndarray,
    isomer_ids: np.ndarray,
) -> tuple[float, int]:
    """Fraction of enantiomer pairs ranked correctly by predicted ``top_score``.

    All four arrays are per-conformer and aligned (the test fold). Conformer
    scores are pooled per stereoisomer; stereoisomers are then grouped by their
    constitution (``molecule_id``). Only constitutions with **exactly two**
    enantiomers are scored (the dataset guarantees pairs; anything else is an
    ingest anomaly and is skipped). A pair counts 1.0 when the predicted lower
    score lands on the truly-lower-scoring enantiomer, 0.0 when reversed, and 0.5
    on a predicted tie. Returns ``(accuracy, n_pairs)``; accuracy is NaN when no
    scorable pair exists.
    """
    true_scores = np.asarray(true_scores, dtype=float)
    pred_scores = np.asarray(pred_scores, dtype=float)
    molecule_ids = np.asarray(molecule_ids)
    isomer_ids = np.asarray(isomer_ids)

    true_per_iso = _aggregate_per_isomer(true_scores, isomer_ids)
    pred_per_iso = _aggregate_per_isomer(pred_scores, isomer_ids)

    # One representative molecule_id per isomer (constant within an isomer group).
    iso_to_mol: dict[int, int] = {}
    for iso, mol in zip(isomer_ids, molecule_ids, strict=True):
        iso_to_mol[int(iso)] = int(mol)

    mol_to_isos: dict[int, list[int]] = {}
    for iso, mol in iso_to_mol.items():
        mol_to_isos.setdefault(mol, []).append(iso)

    scores: list[float] = []
    for isos in mol_to_isos.values():
        if len(isos) != 2:
            continue
        a, b = isos
        true_diff = true_per_iso[a] - true_per_iso[b]
        if true_diff == 0.0:
            # No ground-truth winner (margin filtering should prevent this).
            continue
        pred_diff = pred_per_iso[a] - pred_per_iso[b]
        if pred_diff == 0.0:
            scores.append(0.5)
        else:
            scores.append(1.0 if (pred_diff < 0) == (true_diff < 0) else 0.0)

    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


__all__ = ["pair_ranking_accuracy"]
