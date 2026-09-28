"""Record types flowing between pipeline stages.

Two kinds of record leave the pipeline:

``Structure``
    A molecule that is usable for training, carrying its geometry and the full
    provenance of how it got there.
``RejectedRecord``
    A molecule that did not make it, carrying the stage and reason. A rejection
    is either ``broken`` (erroneous and not fixable) or ``filtered``
    (deliberately excluded, e.g. a duplicate or a manual delete correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .taxonomy import CLASS_ORDER

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

Coordinate = tuple[float, float, float]

# Disposition of a rejected record.
BROKEN = "broken"
FILTERED = "filtered"


@dataclass
class Structure:
    """One usable 3D structure with its label and provenance."""

    index: int
    smiles: str  # canonical isomeric SMILES of the emitted molecule
    class_name: str
    label: int
    symbols: list[str]
    coords: list[Coordinate]

    # --- provenance ------------------------------------------------------- #
    source_file: str = ""
    geometry_quality: str = "ok"  # ok | repaired | rebuilt_architector
    correction: str = ""  # e.g. "relabel central->achiral"
    repair_strategy: str = ""  # reembed | strip_salt | strip+reembed | architector
    note: str = ""

    # The live RDKit mol (with explicit hydrogens and the chosen conformer).
    # Carried in memory between stages; never serialised.
    mol: Chem.Mol | None = field(default=None, repr=False, compare=False)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


@dataclass
class RejectedRecord:
    """One molecule that did not reach the final dataset."""

    smiles: str
    class_name: str
    label: int
    source_file: str
    stage: str  # extract | validate | repair | organometallic
    reason: str
    disposition: str  # broken | filtered
    detail: str = ""
    index: int | None = None  # set only once a structure had been indexed

    # Present only when the rejected molecule still has a usable geometry.
    symbols: list[str] = field(default_factory=list)
    coords: list[Coordinate] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def has_geometry(self) -> bool:
        return bool(self.symbols) and len(self.symbols) == len(self.coords)


@dataclass
class CorrectionRecord:
    """One realised (or unmatched) manual label correction."""

    smiles: str
    action: str  # relabel | delete | unmatched
    from_class: str
    to_class: str
    confidence: str
    reason: str
    matched: bool


@dataclass
class StereoAuditRecord:
    """Per-molecule result of the 3D stereocentre audit of the central class."""

    smiles: str
    source_file: str
    assigned_rs_3d: int
    tetrahedral_tagged_3d: int
    error: str = ""

    @property
    def has_stereocenter(self) -> bool:
        return self.assigned_rs_3d > 0 or self.tetrahedral_tagged_3d > 0


@dataclass
class BuildResult:
    """Everything one pipeline run produced."""

    structures: list[Structure] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    stereo_audit: list[StereoAuditRecord] = field(default_factory=list)
    uncovered_mislabels: list[str] = field(default_factory=list)
    stage_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_ORDER}
        for structure in self.structures:
            counts[structure.class_name] += 1
        return counts

    @property
    def rejected_broken(self) -> list[RejectedRecord]:
        return [r for r in self.rejected if r.disposition == BROKEN]

    @property
    def rejected_filtered(self) -> list[RejectedRecord]:
        return [r for r in self.rejected if r.disposition == FILTERED]
