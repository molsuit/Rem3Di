"""The ``smiles`` -> ``conformers`` row expansion of §1.2.

The transition **changes the row set**: stereoisomers that failed to embed are
gone and the survivors are multiplied by their conformer count. This module
does the bookkeeping only — it never embeds anything. The caller (the
``generate_conformers`` prepare task, build-order step 3) supplies the frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ase import Atoms

from remedi.data_handling.bundle.bundle import (
    Bundle,
    BundleValidationError,
    content_hash_of_table,
)
from remedi.data_handling.bundle.provenance import ConformerGenerationRecord
from remedi.data_handling.bundle.spec import GeometryOrigin
from remedi.data_handling.bundle.validate import validate_bundle

#: ``counts.dropped`` key for a stereoisomer that produced no frame at all.
CONFORMER_EMBEDDING_FAILED = "conformer_embedding_failed"
#: ``counts.dropped`` key for §1.2's orphan rule.
ENANTIOMER_PARTNER_FAILED = "enantiomer_partner_failed"


@dataclass(frozen=True)
class ExpandedBundle:
    """The result of :func:`expand_to_conformers`."""

    bundle: Bundle
    #: reason -> count, for ``provenance.counts.dropped``. Already merged into
    #: ``bundle.provenance`` as well; returned separately so the caller can log it.
    dropped_counts: dict[str, int] = field(default_factory=dict)


def _apply_drop_rules(
    source: pd.DataFrame,
    structures_by_stereoisomer: Mapping[int, Sequence[Atoms]],
    *,
    require_enantiomer_pairs: bool,
) -> tuple[set[int], dict[str, int]]:
    """Which stereoisomers survive the stage boundary, and why the others did not.

    A stereoisomer with no frame failed to embed. A survivor whose mirror
    partner failed is an *orphan*: §1.2 nullifies its pointer, and drops it too
    when the bundle declares ``require_enantiomer_pairs``. Either way the case
    is counted as ``enantiomer_partner_failed``.
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


def expand_to_conformers(
    bundle_smiles: Bundle,
    structures_by_stereoisomer: Mapping[int, Sequence[Atoms]],
    *,
    geometry_origin: GeometryOrigin = "etkdg_mmff",
    conformer_record: ConformerGenerationRecord | None = None,
) -> ExpandedBundle:
    """Expand a ``smiles``-stage bundle into a ``conformers``-stage bundle.

    ``structures_by_stereoisomer`` maps a ``stereoisomer_id`` to the frames
    generated for it; a stereoisomer that is absent or maps to an empty
    sequence is dropped. Per §1.2, ``structure_id`` is re-assigned densely,
    the four identity columns carry through, labels / splits / extras are
    gathered onto the expanded rows by ``stereoisomer_id``, and the orphan rule
    applies: an enantiomer whose partner failed gets a null ``enantiomer_of``,
    or is dropped as well when the spec declares ``require_enantiomer_pairs``.

    Raises:
        ValueError: if the input is not a valid one-row-per-stereoisomer
            ``smiles``-stage bundle.
        BundleValidationError: if the expanded bundle violates §1.1.
    """
    spec = bundle_smiles.spec
    if spec.stage != "smiles":
        raise ValueError(
            f"expand_to_conformers needs a smiles-stage bundle, got {spec.stage!r}"
        )
    source = bundle_smiles.table
    if source["stereoisomer_id"].duplicated().any():
        raise ValueError(
            "a smiles-stage bundle must hold one row per stereoisomer, but "
            "stereoisomer_id repeats"
        )

    surviving, dropped_counts = _apply_drop_rules(
        source,
        structures_by_stereoisomer,
        require_enantiomer_pairs=spec.require_enantiomer_pairs,
    )

    # Materialise the expanded rows and frames in source order.
    source_row_positions: list[int] = []
    structures: list[Atoms] = []
    for row_position, stereoisomer_id in enumerate(source["stereoisomer_id"]):
        if int(stereoisomer_id) not in surviving:
            continue
        for atoms in structures_by_stereoisomer[int(stereoisomer_id)]:
            frame = atoms.copy()
            frame.info = dict(atoms.info)
            frame.info["structure_id"] = len(structures)
            structures.append(frame)
            source_row_positions.append(row_position)

    table = source.iloc[source_row_positions].reset_index(drop=True)
    table["structure_id"] = np.arange(len(table), dtype="int64")
    table["enantiomer_of"] = (
        table["enantiomer_of"]
        .where(table["enantiomer_of"].isin(sorted(surviving)), pd.NA)
        .astype("Int64")
    )

    # 4. Spec and provenance for the far side of the boundary.
    expanded_spec = spec.model_copy(
        update={"stage": "conformers", "geometry_origin": geometry_origin}
    )
    provenance = bundle_smiles.provenance.model_copy(deep=True)
    provenance.outputs = None
    merged_dropped = dict(provenance.counts.dropped)
    for reason, count in dropped_counts.items():
        merged_dropped[reason] = merged_dropped.get(reason, 0) + count
    provenance.counts.dropped = merged_dropped
    if conformer_record is not None:
        record = conformer_record.model_copy(deep=True)
        if record.parent_bundle_content_sha256 is None:
            record.parent_bundle_content_sha256 = content_hash_of_table(source)
        provenance.conformers = record

    expanded = Bundle(
        spec=expanded_spec,
        table=table,
        structures=structures,
        provenance=provenance,
    )
    problems = validate_bundle(expanded)
    if problems:
        raise BundleValidationError(None, problems)
    return ExpandedBundle(bundle=expanded, dropped_counts=dropped_counts)
