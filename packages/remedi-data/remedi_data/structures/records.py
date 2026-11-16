"""What a structure source yields, and how one record becomes a kept frame or a drop.

A source reads its raw files and yields one :class:`StructureRecord` per
structure, or a :class:`DroppedRecord` for one it could not read. The pipeline
turns each record into a :class:`PreparedFrame` or a drop reason with
:func:`prepare_record`; nothing here knows which source a record came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from remedi_data.bundle import mirror_isomeric_smiles, stereo_agrees_with_frame
from remedi_data.chemistry.charge import charge_and_multiplicity
from remedi_data.chemistry.geometry import GeometryLimits, frame_geometry_violations
from remedi_data.chemistry.smiles_filter import (
    SmilesFilterConfig,
    standardized_smiles_for_structure,
)
from remedi_data.dataset.packed_structures import (
    ATOMIC_NUMBER_DTYPE,
    POSITION_DTYPE,
)

#: ``counts.dropped`` keys the pipeline itself uses; geometry limits add theirs
#: (``max_atoms``, ``element_gate``, ...) and sources add their own read errors.
UNREADABLE = "unreadable"
STEREO_FROM_GEOMETRY_FAILED = "stereo_from_geometry_failed"
SMILES_FILTER = "smiles_filter"
SMILES_DOES_NOT_MATCH_STRUCTURE = "smiles_does_not_match_structure"
STEREO_MISMATCH = "stereo_mismatch"
UNSTABLE_CANONICAL_SMILES = "unstable_canonical_smiles"
SOURCE_SMILES_UNPARSABLE = "source_smiles_unparsable"
CONNECTIVITY_DIFFERS_FROM_SOURCE = "connectivity_differs_from_source"

#: Round trips after which a canonical SMILES that still changes is dropped.
_MAXIMUM_ROUND_TRIPS = 3


def round_trip_canonical_smiles(smiles: str) -> tuple[str, Chem.Mol] | None:
    """The canonical SMILES that re-parsing no longer changes, with its molecule.

    A SMILES written from a molecule perceived from 3D is not always a fixed
    point: for some achiral cages (2-substituted adamantanes in PCQM4Mv2)
    parsing and writing it again gives another spelling of the stereo tags.
    The table must hold the fixed point, or one molecule gets two
    ``stereoisomer_id`` values depending on where its SMILES came from.
    Returns ``None`` if it is unparsable or still changes after a few round trips.
    """
    for _ in range(_MAXIMUM_ROUND_TRIPS):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        rewritten = Chem.MolToSmiles(molecule)
        if rewritten == smiles:
            return smiles, molecule
        smiles = rewritten
    return None


@dataclass
class StructureRecord:
    """One structure as a source read it.

    A source that derives SMILES from its geometry sets ``molecule``: an RDKit
    molecule with explicit hydrogens and one conformer, whose formal charges
    and radicals give the total charge and multiplicity. A source without
    SMILES sets ``atomic_numbers``, ``positions``, ``total_charge`` and
    ``multiplicity`` instead.

    ``graph_smiles`` is a SMILES the source vouches for (GEOM's summary key).
    When set, ``molecule`` contributes only its connectivity and coordinates:
    bond orders and charges come from ``graph_smiles``, and a frame whose
    connectivity (hydrogens included) differs from it is dropped. Stereo is
    still perceived from the coordinates.
    """

    source_key: str
    molecule: Chem.Mol | None = None
    graph_smiles: str | None = None
    atomic_numbers: np.ndarray | None = None
    positions: np.ndarray | None = None
    total_charge: float | None = None
    multiplicity: float | None = None
    #: label name -> value; a missing name is a missing label (NaN).
    labels: dict[str, float] = field(default_factory=dict)
    #: extra column name -> value.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DroppedRecord:
    """A record the source could not turn into a :class:`StructureRecord`."""

    source_key: str
    reason: str


@dataclass
class PreparedFrame:
    """A kept structure: the frame, its physics and, for SMILES sources, its identity.

    For a SMILES source the atoms are in the RDKit ``AddHs`` order of
    ``isomeric_smiles``, which is what invariant 9 compares against.
    """

    source_key: str
    atomic_numbers: np.ndarray
    positions: np.ndarray
    total_charge: float
    multiplicity: float
    labels: dict[str, float]
    extras: dict[str, Any]
    isomeric_smiles: str | None = None
    nonisomeric_smiles: str | None = None
    #: Canonical SMILES of the exact mirror image (equal to ``isomeric_smiles``
    #: for a molecule that is its own mirror image).
    mirror_smiles: str | None = None


def _element_graph(molecule: Chem.Mol) -> Chem.Mol:
    """``molecule`` reduced to elements and connectivity: single bonds, no charges."""
    graph = Chem.RWMol(molecule)
    for atom in graph.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumRadicalElectrons(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in graph.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    reduced = graph.GetMol()
    reduced.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(reduced)
    return reduced


def molecule_on_source_graph(molecule: Chem.Mol, graph_smiles: str) -> Chem.Mol | str:
    """The molecule of ``graph_smiles`` (with hydrogens) carrying ``molecule``'s coordinates.

    Atoms are matched on elements and connectivity only, so a frame whose bond
    orders or charges were perceived differently still maps, while one whose
    connectivity changed (a ring closed, a hydrogen moved) does not. Returns
    the ``counts.dropped`` reason when it cannot be mapped.
    """
    parsed = Chem.MolFromSmiles(graph_smiles)
    if parsed is None:
        return SOURCE_SMILES_UNPARSABLE
    template = Chem.AddHs(parsed)
    if (
        template.GetNumAtoms() != molecule.GetNumAtoms()
        or template.GetNumBonds() != molecule.GetNumBonds()
        or molecule.GetNumConformers() != 1
    ):
        return CONNECTIVITY_DIFFERS_FROM_SOURCE
    match = _element_graph(molecule).GetSubstructMatch(_element_graph(template))
    if len(match) != template.GetNumAtoms():
        return CONNECTIVITY_DIFFERS_FROM_SOURCE
    source_positions = molecule.GetConformer().GetPositions()
    conformer = Chem.Conformer(template.GetNumAtoms())
    for template_index, source_index in enumerate(match):
        x, y, z = (float(value) for value in source_positions[source_index])
        conformer.SetAtomPosition(template_index, Point3D(x, y, z))
    conformer.Set3D(True)
    template.RemoveAllConformers()
    template.AddConformer(conformer, assignId=True)
    return template


def _frame_from_molecule(
    record: StructureRecord, smiles_filter: SmilesFilterConfig
) -> PreparedFrame | str:
    """SMILES from the geometry, atoms reordered to the SMILES, invariant 9 checked."""
    molecule = record.molecule
    assert molecule is not None
    if molecule.GetNumConformers() != 1:
        return UNREADABLE
    try:
        Chem.AssignStereochemistryFrom3D(molecule)
        implicit_hydrogens = Chem.RemoveAllHs(molecule)
    except (ValueError, RuntimeError):
        return STEREO_FROM_GEOMETRY_FAILED
    # No ``seen``: conformers and repeated molecules are all kept, they share ids.
    isomeric_smiles = standardized_smiles_for_structure(
        implicit_hydrogens, smiles_filter
    )
    if isomeric_smiles is None:
        return SMILES_FILTER
    stable = round_trip_canonical_smiles(isomeric_smiles)
    if stable is None:
        return UNSTABLE_CANONICAL_SMILES
    isomeric_smiles, parsed = stable
    template = Chem.AddHs(parsed)
    # The match fails when standardisation moved a hydrogen or changed the
    # heavy-atom graph, i.e. whenever the SMILES no longer describes the frame.
    # Charges cannot change: the source refuses ``neutralize``, and salt
    # stripping that removes atoms was already dropped by the filter.
    if template.GetNumAtoms() != molecule.GetNumAtoms():
        return SMILES_DOES_NOT_MATCH_STRUCTURE
    match = molecule.GetSubstructMatch(template)
    if len(match) != template.GetNumAtoms():
        return SMILES_DOES_NOT_MATCH_STRUCTURE
    order = np.asarray(match, dtype=np.int64)
    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)[
        order
    ].astype(POSITION_DTYPE)
    atomic_numbers = np.array(
        [atom.GetAtomicNum() for atom in template.GetAtoms()],
        dtype=ATOMIC_NUMBER_DTYPE,
    )
    if not stereo_agrees_with_frame(isomeric_smiles, atomic_numbers, positions):
        return STEREO_MISMATCH
    total_charge, multiplicity = charge_and_multiplicity(template)
    has_tetrahedral_tag = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in parsed.GetAtoms()
    )
    return PreparedFrame(
        source_key=record.source_key,
        atomic_numbers=atomic_numbers,
        positions=positions,
        total_charge=total_charge,
        multiplicity=multiplicity,
        labels=record.labels,
        extras=record.extras,
        isomeric_smiles=isomeric_smiles,
        nonisomeric_smiles=Chem.MolToSmiles(parsed, isomericSmiles=False),
        mirror_smiles=(
            mirror_isomeric_smiles(isomeric_smiles)
            if has_tetrahedral_tag
            else isomeric_smiles
        ),
    )


def _frame_from_arrays(record: StructureRecord) -> PreparedFrame | str:
    if (
        record.atomic_numbers is None
        or record.positions is None
        or record.total_charge is None
        or record.multiplicity is None
        or len(record.atomic_numbers) != len(record.positions)
    ):
        return UNREADABLE
    return PreparedFrame(
        source_key=record.source_key,
        atomic_numbers=np.asarray(record.atomic_numbers, dtype=ATOMIC_NUMBER_DTYPE),
        positions=np.asarray(record.positions, dtype=POSITION_DTYPE).reshape(-1, 3),
        total_charge=float(record.total_charge),
        multiplicity=float(record.multiplicity),
        labels=record.labels,
        extras=record.extras,
    )


def prepare_record(
    record: StructureRecord,
    *,
    smiles_filter: SmilesFilterConfig | None,
    limits: GeometryLimits,
    allowed_atomic_numbers: frozenset[int] | None,
) -> PreparedFrame | str:
    """The kept frame, or the ``counts.dropped`` reason it was dropped for.

    ``smiles_filter`` set means the dataset declares ``smiles: true``: the
    record must carry a molecule, the SMILES is derived from it and the frame
    is reordered to it. A record is counted once, under the first reason.
    """
    if smiles_filter is not None:
        if record.molecule is None:
            return UNREADABLE
        if record.graph_smiles is not None:
            # The filter first, on the trusted SMILES, so a molecule the filter
            # rejects anyway (GEOM's charged ones) is counted under that reason.
            if (
                standardized_smiles_for_structure(
                    Chem.MolFromSmiles(record.graph_smiles), smiles_filter
                )
                is None
            ):
                return SMILES_FILTER
            on_graph = molecule_on_source_graph(record.molecule, record.graph_smiles)
            if isinstance(on_graph, str):
                return on_graph
            record = replace(record, molecule=on_graph)
        try:
            prepared = _frame_from_molecule(record, smiles_filter)
        except (ValueError, RuntimeError):
            return UNREADABLE
    else:
        prepared = _frame_from_arrays(record)
    if isinstance(prepared, str):
        return prepared
    violations = frame_geometry_violations(
        prepared.atomic_numbers, prepared.positions, limits, allowed_atomic_numbers
    )
    return violations[0] if violations else prepared
