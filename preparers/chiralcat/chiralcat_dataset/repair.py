"""Stage 3: detect clashing geometries and repair the ones that can be repaired.

Two repair strategies, each behind a gate that refuses a repair which would
change the molecule's chirality:

``reembed``
    Regenerate the geometry from the isomeric SMILES (ETKDGv3 + MMFF94). Only
    valid where the SMILES fully encodes the chirality (central / achiral).
``strip_salt``
    Keep the largest fragment of the original conformer, dropping a counter-ion
    that overlapped it. No heavy atom moves, so the chirality geometry survives
    by construction.

Anything left clashing is rejected as broken; stage 4 rescues the subset that is
a rebuildable organometallic.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .chemistry import (
    METAL_SYMBOLS,
    SMILES_ENCODED_CLASSES,
    add_hydrogens,
    canonical_smiles,
    categorize_clash,
    chirality_preserved,
    find_clashes,
    has_unspecified_stereo,
    identity_signature,
    mol_to_arrays,
    pick_conformer_id,
    single_conformer,
    standardize_mol,
)
from .config import PipelineConfig
from .records import BROKEN, FILTERED, Coordinate, RejectedRecord, Structure

__all__ = [
    "METAL_SYMBOLS",
    "RepairOutcome",
    "RepairOutput",
    "reembed",
    "repair",
    "repair_structure",
    "strip_counterion",
]

_REEMBED_CATEGORIES = frozenset({"collapsed_geometry_organic", "hydrogen_artifact"})
_STRIP_CATEGORIES = frozenset({"multi_fragment_salt"})

# Best-to-worst ordering used when a repair collapses two entries onto one SMILES.
_QUALITY_RANK = {"ok": 0, "repaired": 1, "poor": 2}


@dataclass
class RepairOutcome:
    """What the repair stage did to one clashing structure."""

    index: int
    class_name: str
    label: int
    category: str
    strategy: str  # reembed | strip_salt | strip+reembed | flag
    status: str  # repaired | flagged | failed_stereo | failed_clash | failed
    old_smiles: str
    new_smiles: str
    old_min_distance: float
    new_min_distance: float | None
    stereo_ok: bool
    note: str = ""
    # populated only when status == "repaired"
    symbols: list[str] = field(default_factory=list)
    coords: list[Coordinate] = field(default_factory=list)


@dataclass
class RepairOutput:
    structures: list[Structure] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    outcomes: list[RepairOutcome] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def reembed(smiles: str, *, seed: int, mmff_iters: int) -> Chem.Mol | None:
    """Regenerate a clean 3D geometry from an isomeric SMILES (ETKDGv3 + MMFF).

    ``AllChem`` is populated from RDKit's C++ bindings at import time, so a type
    checker cannot see these members; the ``ty: ignore`` comments below say so
    rather than disabling the rule for the whole project.
    """
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        return None
    mol = Chem.AddHs(base)
    params = AllChem.ETKDGv3()  # ty: ignore[unresolved-attribute]
    params.randomSeed = seed
    params.enforceChirality = True
    if AllChem.EmbedMolecule(mol, params) != 0:  # ty: ignore[unresolved-attribute]
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:  # ty: ignore[unresolved-attribute]
            return None
    with contextlib.suppress(Exception):
        AllChem.MMFFOptimizeMolecule(  # ty: ignore[unresolved-attribute]
            mol, maxIters=mmff_iters
        )
    return mol


def strip_counterion(source_mol: Chem.Mol, conformer_id: int) -> Chem.Mol | None:
    """Keep the largest fragment of a source conformer, removing counter-ions.

    The kept fragment retains its original coordinates (chirality geometry
    untouched); hydrogens are re-placed so none point at the removed counter-ion.
    """
    working = single_conformer(source_mol, conformer_id)
    fragments = Chem.GetMolFrags(working, asMols=True, sanitizeFrags=False)
    if len(fragments) < 2:
        return None
    cation = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
    with contextlib.suppress(Exception):
        Chem.SanitizeMol(cation)
    cation = standardize_mol(cation, strip_salts=False, neutralize=True)
    if cation is None:
        return None
    with contextlib.suppress(Exception):
        Chem.SanitizeMol(cation)
    return add_hydrogens(cation)


def _apply_clash_gate(outcome: RepairOutcome, repaired: Chem.Mol) -> RepairOutcome:
    symbols, coords = mol_to_arrays(repaired)
    residual = find_clashes(symbols, np.array(coords))
    outcome.new_min_distance = residual[0].distance if residual else None
    if residual:
        outcome.status = "failed_clash"
        outcome.note = f"residual clash {residual[0].distance} A after repair"
        return outcome
    outcome.status = "repaired"
    outcome.symbols = symbols
    outcome.coords = coords
    return outcome


def _finalize_reembed(
    outcome: RepairOutcome, repaired: Chem.Mol, *, intended_smiles: str
) -> RepairOutcome:
    """Re-embed gate: the 3D-perceived chirality must match the SMILES label."""
    preserved, achieved = chirality_preserved(intended_smiles, repaired)
    outcome.stereo_ok = preserved
    outcome.new_smiles = canonical_smiles(intended_smiles) or intended_smiles
    if not preserved:
        outcome.status = "failed_stereo"
        outcome.note = f"chirality changed: got {achieved!r}"
        return outcome
    return _apply_clash_gate(outcome, repaired)


def _finalize_strip(
    outcome: RepairOutcome, repaired: Chem.Mol, *, expected_smiles: str
) -> RepairOutcome:
    """Strip gate: the kept fragment must be exactly the labelled molecule.

    The counter-ion strip never moves a heavy atom, so the chirality geometry is
    preserved by construction. The remaining risk is keeping the wrong fragment
    or perturbing the stereo flags; comparing the repaired molecule's canonical
    isomeric SMILES to the expected cation catches either.
    """
    expected = canonical_smiles(expected_smiles)
    achieved = identity_signature(repaired)
    outcome.new_smiles = expected or expected_smiles
    outcome.stereo_ok = expected is not None and achieved == expected
    if not outcome.stereo_ok:
        outcome.status = "failed_stereo"
        outcome.note = f"fragment/stereo mismatch: got {achieved!r}, want {expected!r}"
        return outcome
    return _apply_clash_gate(outcome, repaired)


def repair_structure(
    structure: Structure,
    config: PipelineConfig,
    source_lookup: dict[str, Chem.Mol],
) -> RepairOutcome:
    """Attempt to repair one clashing structure."""
    coords = np.array(structure.coords)
    clashes = find_clashes(structure.symbols, coords)
    old_min = clashes[0].distance if clashes else float("inf")
    category = categorize_clash(structure.symbols, structure.smiles, clashes)

    outcome = RepairOutcome(
        index=structure.index,
        class_name=structure.class_name,
        label=structure.label,
        category=category,
        strategy="flag",
        status="flagged",
        old_smiles=structure.smiles,
        new_smiles=structure.smiles,
        old_min_distance=old_min,
        new_min_distance=None,
        stereo_ok=False,
    )

    # --- Tier 1: re-embed (SMILES-encoded chirality only) -------------------- #
    if category in _REEMBED_CATEGORIES:
        outcome.strategy = "reembed"
        if structure.class_name not in SMILES_ENCODED_CLASSES:
            outcome.note = (
                f"class {structure.class_name!r} not SMILES-encoded; not re-embedded"
            )
            return outcome
        if has_unspecified_stereo(structure.smiles):
            outcome.note = "SMILES leaves stereochemistry unspecified; cannot re-embed"
            return outcome
        repaired = reembed(
            structure.smiles,
            seed=config.repair.embed_seed,
            mmff_iters=config.repair.mmff_iters,
        )
        if repaired is None:
            outcome.status = "failed"
            outcome.note = "embedding failed"
            return outcome
        return _finalize_reembed(outcome, repaired, intended_smiles=structure.smiles)

    # --- Tier 2: strip counter-ion (keep cation geometry) -------------------- #
    if category in _STRIP_CATEGORIES:
        outcome.strategy = "strip_salt"
        canonical = canonical_smiles(structure.smiles)
        source_mol = source_lookup.get(canonical) if canonical else None
        if source_mol is None:
            outcome.status = "failed"
            outcome.note = "source mol not found for counter-ion strip"
            return outcome
        conformer_id = pick_conformer_id(source_mol, config.extraction.conformer_id)
        repaired = strip_counterion(source_mol, conformer_id)
        if repaired is None:
            outcome.status = "failed"
            outcome.note = "counter-ion strip produced no fragment"
            return outcome
        # Expected molecule = the largest SMILES fragment (the real cation).
        cation_smiles = max(structure.smiles.split("."), key=len)
        stripped = _finalize_strip(outcome, repaired, expected_smiles=cation_smiles)
        if stripped.status != "failed_clash":
            return stripped
        # A residual clash after stripping means the clash is intra-cation (a
        # collapsed group), not a counter-ion overlap. Re-embed the cation if its
        # chirality is SMILES-encoded; otherwise keep the flagged strip outcome.
        if structure.class_name in SMILES_ENCODED_CLASSES and not (
            has_unspecified_stereo(cation_smiles)
        ):
            reembedded = reembed(
                cation_smiles,
                seed=config.repair.embed_seed,
                mmff_iters=config.repair.mmff_iters,
            )
            if reembedded is not None:
                stripped = _finalize_reembed(
                    outcome, reembedded, intended_smiles=cation_smiles
                )
                stripped.strategy = "strip+reembed"
        return stripped

    # --- Out of scope -------------------------------------------------------- #
    outcome.note = "organometallic / duplicate-atom: left unrepaired (flagged)"
    return outcome


def repair(
    config: PipelineConfig,
    structures: list[Structure],
    source_lookup: dict[str, Chem.Mol],
) -> RepairOutput:
    """Run stage 3 over every structure, returning the kept and rejected sets."""
    output = RepairOutput()
    if not config.repair.enabled:
        output.structures = list(structures)
        output.counts = {"ok": len(structures), "skipped": 1}
        return output

    outcomes: dict[int, RepairOutcome] = {}
    for structure in structures:
        if find_clashes(structure.symbols, np.array(structure.coords)):
            outcomes[structure.index] = repair_structure(structure, config, source_lookup)
    output.outcomes = [outcomes[key] for key in sorted(outcomes)]

    # Resolve each structure's final geometry, SMILES and quality.
    resolved: list[tuple[Structure, str]] = []
    for structure in structures:
        outcome = outcomes.get(structure.index)
        if outcome is None:
            resolved.append((structure, "ok"))
            continue
        if outcome.status == "repaired":
            structure.symbols = outcome.symbols
            structure.coords = outcome.coords
            structure.smiles = outcome.new_smiles
            structure.geometry_quality = "repaired"
            structure.repair_strategy = outcome.strategy
            structure.mol = None  # geometry replaced; stale mol would mislead
            resolved.append((structure, "repaired"))
            continue
        structure.repair_strategy = outcome.strategy
        structure.note = outcome.note
        resolved.append((structure, "poor"))

    # A stripped salt can collapse onto an existing free base, exactly as the
    # extraction's own dedup does. Keep the best-quality geometry per SMILES.
    best: dict[str, int] = {}
    for position, (structure, quality) in enumerate(resolved):
        if structure.smiles not in best:
            best[structure.smiles] = position
            continue
        incumbent, incumbent_quality = resolved[best[structure.smiles]]
        if (_QUALITY_RANK[quality], structure.index) < (
            _QUALITY_RANK[incumbent_quality],
            incumbent.index,
        ):
            best[structure.smiles] = position
    kept_positions = set(best.values())
    kept_index_for = {
        smiles: resolved[position][0].index for smiles, position in best.items()
    }

    for position, (structure, quality) in enumerate(resolved):
        if position not in kept_positions:
            output.rejected.append(
                RejectedRecord(
                    index=structure.index,
                    smiles=structure.smiles,
                    class_name=structure.class_name,
                    label=structure.label,
                    source_file=structure.source_file,
                    stage="repair",
                    reason="dropped_duplicate",
                    disposition=FILTERED,
                    detail=(
                        "collapses onto kept index "
                        f"{kept_index_for[structure.smiles]} after repair"
                    ),
                    symbols=structure.symbols,
                    coords=structure.coords,
                )
            )
            continue
        if quality == "poor":
            outcome = outcomes[structure.index]
            output.rejected.append(
                RejectedRecord(
                    index=structure.index,
                    smiles=structure.smiles,
                    class_name=structure.class_name,
                    label=structure.label,
                    source_file=structure.source_file,
                    stage="repair",
                    reason=outcome.status,
                    disposition=BROKEN,
                    detail=f"{outcome.category}: {outcome.note}",
                    symbols=structure.symbols,
                    coords=structure.coords,
                )
            )
            continue
        output.structures.append(structure)

    output.counts = {
        "clashing": len(outcomes),
        "repaired": sum(1 for o in outcomes.values() if o.status == "repaired"),
        "unrepaired": sum(
            1 for record in output.rejected if record.disposition == BROKEN
        ),
        "dropped_duplicate": sum(
            1 for record in output.rejected if record.reason == "dropped_duplicate"
        ),
        "kept": len(output.structures),
    }
    return output
