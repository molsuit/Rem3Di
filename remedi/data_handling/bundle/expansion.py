"""From a bundle (one row per stereoisomer) to a dataset table (one row per structure).

Conformer generation changes the row set: stereoisomers that produced no
structure are gone and the survivors are multiplied by their structure count.
This module does the bookkeeping only; it never embeds anything. The caller
(``dataset_build``) supplies the structures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ase import Atoms

from remedi.data_handling.bundle.bundle import Bundle
from remedi.data_handling.bundle.provenance import BundleProvenance
from remedi.data_handling.bundle.spec import DatasetSpec, GeometryOrigin

#: ``counts.dropped`` key for a stereoisomer that produced no structure at all.
CONFORMER_EMBEDDING_FAILED = "conformer_embedding_failed"
#: ``counts.dropped`` key for the orphan rule (§1.2).
ENANTIOMER_PARTNER_FAILED = "enantiomer_partner_failed"


@dataclass(frozen=True)
class ExpandedDataset:
    """What :func:`expand_to_structures` returns: everything ``write_dataset`` needs."""

    spec: DatasetSpec
    table: pd.DataFrame
    structures: list[Atoms]
    provenance: BundleProvenance
    #: reason -> count, also merged into ``provenance.counts.dropped``.
    dropped_counts: dict[str, int] = field(default_factory=dict)


def _apply_drop_rules(
    source: pd.DataFrame,
    structures_by_stereoisomer: Mapping[int, Sequence[Atoms]],
    *,
    require_enantiomer_pairs: bool,
) -> tuple[set[int], dict[str, int]]:
    """Which stereoisomers survive, and why the others did not.

    A stereoisomer with no structure failed to embed. A survivor whose mirror
    partner failed is an *orphan*: its pointer is nulled, and it is dropped too
    when the evaluation declares ``require_enantiomer_pairs``. Either way the
    case is counted as ``enantiomer_partner_failed``.
    """
    dropped_counts: dict[str, int] = {}
    surviving = {
        int(stereoisomer_id)
        for stereoisomer_id in source["stereoisomer_id"]
        if len(structures_by_stereoisomer.get(int(stereoisomer_id), ())) > 0
    }
    embedding_failures = len(source) - len(surviving)
    if embedding_failures:
        dropped_counts[CONFORMER_EMBEDDING_FAILED] = embedding_failures
    partner_of = source.set_index("stereoisomer_id")["enantiomer_of"]
    orphaned = {
        stereoisomer_id
        for stereoisomer_id in surviving
        if not pd.isna(partner_of[stereoisomer_id])
        and int(partner_of[stereoisomer_id]) not in surviving
    }
    if orphaned:
        dropped_counts[ENANTIOMER_PARTNER_FAILED] = len(orphaned)
    if require_enantiomer_pairs:
        surviving -= orphaned
    return surviving, dropped_counts


def expand_to_structures(
    bundle: Bundle,
    structures_by_stereoisomer: Mapping[int, Sequence[Atoms]],
    charges_by_stereoisomer: Mapping[int, tuple[float, float]],
    *,
    geometry_origin: GeometryOrigin,
) -> ExpandedDataset:
    """Expand a bundle into a dataset table plus row-aligned structures.

    ``structures_by_stereoisomer`` maps a ``stereoisomer_id`` to its structures;
    an absent or empty entry drops the stereoisomer. ``charges_by_stereoisomer``
    gives ``(total_charge, multiplicity)`` for every surviving stereoisomer.
    ``structure_id`` is assigned densely, the identity columns carry through,
    labels / splits / extras are repeated onto every structure of their
    stereoisomer, and the orphan rule applies.

    Raises:
        ValueError: if ``bundle`` is not a bundle (it already has structures)
            or a surviving stereoisomer has no charge entry.
    """
    spec = bundle.spec
    if spec.has_structures:
        raise ValueError(
            f"{spec.dataset_id} already has structures; expand a bundle, not a dataset"
        )
    source = bundle.table
    require_pairs = (
        spec.evaluation is not None and spec.evaluation.require_enantiomer_pairs
    )
    surviving, dropped_counts = _apply_drop_rules(
        source, structures_by_stereoisomer, require_enantiomer_pairs=require_pairs
    )
    missing_charges = sorted(surviving - set(charges_by_stereoisomer))
    if missing_charges:
        raise ValueError(
            f"no (total_charge, multiplicity) for stereoisomers {missing_charges[:5]}"
        )

    source_row_positions: list[int] = []
    structures: list[Atoms] = []
    for row_position, stereoisomer_id in enumerate(source["stereoisomer_id"]):
        if int(stereoisomer_id) not in surviving:
            continue
        for atoms in structures_by_stereoisomer[int(stereoisomer_id)]:
            structures.append(atoms)
            source_row_positions.append(row_position)

    table = source.iloc[source_row_positions].reset_index(drop=True)
    table.insert(0, "structure_id", np.arange(len(table), dtype="int64"))
    table["enantiomer_of"] = (
        table["enantiomer_of"]
        .where(table["enantiomer_of"].isin(sorted(surviving)), pd.NA)
        .astype("Int64")
    )
    charges = [
        charges_by_stereoisomer[int(stereoisomer_id)]
        for stereoisomer_id in table["stereoisomer_id"]
    ]
    table["total_charge"] = np.array([charge for charge, _ in charges], dtype="float64")
    table["multiplicity"] = np.array(
        [multiplicity for _, multiplicity in charges], dtype="float64"
    )

    expanded_spec = spec.with_structures(geometry_origin)
    table = table[expanded_spec.expected_columns()]
    provenance = bundle.provenance.model_copy(deep=True)
    provenance.outputs = None
    merged = dict(provenance.counts.dropped)
    for reason, count in dropped_counts.items():
        merged[reason] = merged.get(reason, 0) + count
    provenance.counts.dropped = merged
    return ExpandedDataset(
        spec=expanded_spec,
        table=table,
        structures=structures,
        provenance=provenance,
        dropped_counts=dropped_counts,
    )
