"""What a structure source yields, and how one record becomes a kept frame or a drop.

A source reads its raw files and yields one :class:`StructureRecord` per
structure, or a :class:`DroppedRecord` for one it could not read. The pipeline
turns each record into a :class:`PreparedFrame` or a drop reason with
:func:`prepare_record`; nothing here knows which source a record came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem

from remedi_data.bundle.identity import mirror_isomeric_smiles
from remedi_data.bundle.validate import stereo_agrees_with_frame
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
CHARGE_CHANGED_BY_STANDARDIZATION = "charge_changed_by_standardization"
STEREO_MISMATCH = "stereo_mismatch"


@dataclass
class StructureRecord:
    """One structure as a source read it.

    A source that derives SMILES from its geometry sets ``molecule``: an RDKit
    molecule with explicit hydrogens and one conformer, whose formal charges
    and radicals give the total charge and multiplicity. A source without
    SMILES sets ``atomic_numbers``, ``positions``, ``total_charge`` and
    ``multiplicity`` instead.
    """

    source_key: str
    molecule: Chem.Mol | None = None
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
    parsed = Chem.MolFromSmiles(isomeric_smiles)
    if parsed is None:
        return SMILES_FILTER
    template = Chem.AddHs(parsed)
    # The match fails when standardisation moved a hydrogen or changed the
    # heavy-atom graph, i.e. whenever the SMILES no longer describes the frame.
    if template.GetNumAtoms() != molecule.GetNumAtoms():
        return SMILES_DOES_NOT_MATCH_STRUCTURE
    match = molecule.GetSubstructMatch(template)
    if len(match) != template.GetNumAtoms():
        return SMILES_DOES_NOT_MATCH_STRUCTURE
    total_charge = float(Chem.GetFormalCharge(template))
    if total_charge != float(Chem.GetFormalCharge(molecule)):
        return CHARGE_CHANGED_BY_STANDARDIZATION
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
    radical_electrons = sum(
        atom.GetNumRadicalElectrons() for atom in template.GetAtoms()
    )
    has_tetrahedral_tag = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in parsed.GetAtoms()
    )
    return PreparedFrame(
        source_key=record.source_key,
        atomic_numbers=atomic_numbers,
        positions=positions,
        total_charge=total_charge,
        multiplicity=float(radical_electrons + 1),
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
