"""Stage 4: restore the flagged organometallic complexes from frozen rebuilds.

The source pickles store coordinated sandwich complexes as disconnected
fragments drawn without the metal bonds, so their conformers have the metal
sitting on top of the ring carbons. Those geometries cannot be repaired by
re-embedding (RDKit has no bonds to work with). Instead the fragments were
translated back into a metal core plus haptic ligands and each complex was
assembled once with Architector and relaxed with xtb. Those geometries are
frozen in one extxyz file keyed by the fragmented source SMILES; this stage
looks every target up there and re-validates the geometry on composition,
hapticity and clashes before it ships.

Three scaffold families occur in the data:
    ferrocene       -> Fe(II) sandwich, two Cp- rings
    (arene)Cr(CO)3  -> Cr(0) piano stool
    (Cp)Mn(CO)3     -> Mn(I) piano stool (cymantrene)
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem

from .chemistry import find_clashes
from .config import PipelineConfig
from .records import BROKEN, Coordinate, RejectedRecord, Structure
from .sources import verify_sha256

# The metals of the scaffold families present in the dataset.
SUPPORTED_METALS: tuple[str, ...] = ("Fe", "Cr", "Mn")

CARBON_MONOXIDE_SMILES = "[C-]#[O+]"

# The haptic shell around the metal, in Angstrom.
_HAPTIC_SHELL = (2.0, 2.5)


# --------------------------------------------------------------------------- #
# Ligand translation: radical ring fragment -> aromatic sandwich ligand
# --------------------------------------------------------------------------- #


@dataclass
class SandwichLigand:
    """A haptic ring ligand: aromatic SMILES plus its coordinating face."""

    smiles: str
    coord_list: list[int]


def _carbocyclic_face(mol: Chem.Mol) -> tuple[int, ...] | None:
    """Return the atom indices of a 5- or 6-membered all-carbon ring, if any."""
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) in (5, 6) and all(
            mol.GetAtomWithIdx(index).GetSymbol() == "C" for index in ring
        ):
            return ring
    return None


def ring_to_sandwich(fragment: str) -> SandwichLigand | None:
    """Convert a radical-form ring fragment into an aromatic sandwich ligand.

    The source stores coordinated rings as carbene/radical SMILES drawn without
    the metal bonds (e.g. ``C[C]1[CH][CH][CH][CH][C]1C=O``). Aromatizing the ring
    (Cp- anion for 5-membered, neutral arene for 6-membered) makes RDKit accept
    it, preserving every substituent and its ring position.
    """
    mol = Chem.MolFromSmiles(fragment, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    ring = _carbocyclic_face(mol)
    if ring is None:
        return None

    ring_set = set(ring)
    editable = Chem.RWMol(mol)
    for index in ring:
        atom = editable.GetAtomWithIdx(index)
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(True)
    for bond in editable.GetBonds():
        if bond.GetBeginAtomIdx() in ring_set and bond.GetEndAtomIdx() in ring_set:
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
    if len(ring) == 5:
        # Cyclopentadienyl anion: place the -1 on an unsubstituted ring CH so the
        # 6-pi-electron aromatic system kekulizes.
        for index in ring:
            atom = editable.GetAtomWithIdx(index)
            exocyclic_heavy = [
                neighbor
                for neighbor in atom.GetNeighbors()
                if neighbor.GetIdx() not in ring_set and neighbor.GetSymbol() != "H"
            ]
            if not exocyclic_heavy:
                atom.SetFormalCharge(-1)
                break

    aromatized = editable.GetMol()
    try:
        Chem.SanitizeMol(aromatized)
    except Exception:
        return None
    smiles = Chem.MolToSmiles(aromatized)

    # Re-perceive the face on the canonical SMILES the ligand is recorded as.
    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None:
        return None
    face = _carbocyclic_face(reparsed)
    if face is None:
        return None
    return SandwichLigand(smiles=smiles, coord_list=list(face))


# --------------------------------------------------------------------------- #
# Complex parsing
# --------------------------------------------------------------------------- #


@dataclass
class ComplexSpec:
    metal: str
    n_carbon_monoxide: int
    ligands: list[SandwichLigand]

    @property
    def family(self) -> str:
        if self.n_carbon_monoxide == 3:
            return f"{self.metal}(CO)3 piano-stool"
        if self.metal == "Fe" and self.n_carbon_monoxide == 0:
            return "ferrocene (bis-Cp)"
        return f"{self.metal} other"

    @property
    def component_smiles(self) -> str:
        """The complex as dot-separated components: ligands, carbonyls, metal.

        An identity key, not a bonded description: SMILES has no notation for a
        haptic bond. It must name the metal and the carbonyls, or an (arene)Cr(CO)3
        and an (arene)Mn(CO)3 of the same arene share a SMILES.
        """
        components = [ligand.smiles for ligand in self.ligands]
        components += [CARBON_MONOXIDE_SMILES] * self.n_carbon_monoxide
        components.append(f"[{self.metal}]")
        return ".".join(components)

    def expected_counts(self) -> dict[str, int]:
        """Atom counts of the assembled neutral complex, for validation."""
        counts: dict[str, int] = {self.metal: 1}
        if self.n_carbon_monoxide:  # each CO contributes one C and one O
            counts["C"] = counts.get("C", 0) + self.n_carbon_monoxide
            counts["O"] = counts.get("O", 0) + self.n_carbon_monoxide
        for ligand in self.ligands:
            mol = Chem.AddHs(Chem.MolFromSmiles(ligand.smiles))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                counts[symbol] = counts.get(symbol, 0) + 1
        return counts


def _metal_of(fragments: list[str]) -> str | None:
    for symbol in SUPPORTED_METALS:
        if any(symbol in fragment for fragment in fragments):
            return symbol
    return None


def parse_complex(smiles: str) -> ComplexSpec | None:
    """Split a fragmented organometallic SMILES into a buildable ComplexSpec."""
    fragments = smiles.split(".")
    metal = _metal_of(fragments)
    if metal is None:
        return None
    n_carbon_monoxide = sum(
        1 for fragment in fragments if fragment == CARBON_MONOXIDE_SMILES
    )
    organic = [
        fragment
        for fragment in fragments
        if fragment != CARBON_MONOXIDE_SMILES
        and not (
            set(fragment) <= set("[]+0123456789")
            or (metal in fragment and len(fragment) <= 6)
        )
    ]
    ligands: list[SandwichLigand] = []
    for fragment in organic:
        ligand = ring_to_sandwich(fragment)
        if ligand is None:
            return None
        ligands.append(ligand)
    if not ligands:
        return None
    return ComplexSpec(metal=metal, n_carbon_monoxide=n_carbon_monoxide, ligands=ligands)


# --------------------------------------------------------------------------- #
# Frozen rebuilt geometries
# --------------------------------------------------------------------------- #


@dataclass
class FrozenRebuild:
    """One Architector-built complex, read back from the frozen extxyz file."""

    source_smiles: str
    symbols: list[str]
    coords: list[Coordinate]


def read_frozen_rebuilds(path: Path) -> dict[str, FrozenRebuild]:
    """Read the frozen rebuilds, keyed by the fragmented source SMILES.

    Each frame's comment line carries ``source_smiles``; a repeated key or a
    frame whose atom block is short fails the run rather than shipping a
    truncated complex.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    rebuilds: dict[str, FrozenRebuild] = {}
    position = 0
    while position < len(lines):
        n_atoms = int(lines[position])
        fields = dict(token.split("=", 1) for token in shlex.split(lines[position + 1]))
        atom_lines = lines[position + 2 : position + 2 + n_atoms]
        if len(atom_lines) != n_atoms:
            raise ValueError(f"{path}: truncated frame at line {position + 1}")
        source_smiles = fields.get("source_smiles")
        if not source_smiles:
            raise ValueError(f"{path}: frame at line {position + 1} has no source_smiles")
        if source_smiles in rebuilds:
            raise ValueError(f"{path}: duplicate source_smiles {source_smiles}")
        symbols: list[str] = []
        coords: list[Coordinate] = []
        for atom_line in atom_lines:
            symbol, x, y, z = atom_line.split()
            symbols.append(symbol)
            coords.append((float(x), float(y), float(z)))
        rebuilds[source_smiles] = FrozenRebuild(source_smiles, symbols, coords)
        position += 2 + n_atoms
    return rebuilds


# --------------------------------------------------------------------------- #
# Validate
# --------------------------------------------------------------------------- #


@dataclass
class RebuildOutput:
    structures: list[Structure] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def _haptic_ring_carbons(symbols: list[str], coords: np.ndarray, metal: str) -> int:
    """Count carbons in the haptic shell of the metal."""
    metal_index = symbols.index(metal)
    distances = np.linalg.norm(coords - coords[metal_index], axis=1)
    low, high = _HAPTIC_SHELL
    return int(
        sum(
            1
            for i, symbol in enumerate(symbols)
            if symbol == "C" and low < distances[i] < high
        )
    )


def validate_rebuild(spec: ComplexSpec, rebuild: FrozenRebuild) -> str | None:
    """Check a rebuilt geometry against its complex; return why it fails, if it does.

    The gates are the ones the Architector build was accepted on: the atoms are
    exactly the intended neutral complex, every ring carbon of every haptic face
    sits in the metal's haptic shell, and nothing clashes.
    """
    coords = np.asarray(rebuild.coords, dtype=float)

    observed: dict[str, int] = {}
    for symbol in rebuild.symbols:
        observed[symbol] = observed.get(symbol, 0) + 1
    expected = {key: value for key, value in spec.expected_counts().items() if value}
    if observed != expected:
        return f"composition {observed} != expected {expected}"

    face_size = sum(len(ligand.coord_list) for ligand in spec.ligands)
    haptic = _haptic_ring_carbons(rebuild.symbols, coords, spec.metal)
    if haptic < face_size:
        return f"only {haptic}/{face_size} ring C haptically bound"

    clashes = find_clashes(rebuild.symbols, coords)
    if clashes:
        return f"{len(clashes)} residual clash(es)"
    return None


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #


def _reject(record: RejectedRecord, status: str, detail: str = "") -> RejectedRecord:
    note = f"rebuild {status}: {detail}" if detail else f"rebuild {status}"
    record.detail = f"{record.detail}; {note}" if record.detail else note
    return record


def rebuild(
    config: PipelineConfig,
    rejected: list[RejectedRecord],
    progress=None,
) -> RebuildOutput:
    """Restore every broken organometallic of the configured class.

    Takes the repair stage's broken records and returns the ones with a valid
    frozen rebuild as structures, alongside the records that stay rejected.
    Raises ``FileNotFoundError`` if the frozen file is missing and
    ``ChecksumMismatchError`` if it is not the pinned one.
    """
    output = RebuildOutput()
    if not config.organometallic.enabled:
        output.rejected = list(rejected)
        return output

    path = config.rebuilt_structures_path
    if not path.is_file():
        raise FileNotFoundError(f"Frozen organometallic rebuilds not found: {path}")
    verify_sha256(path, config.organometallic.rebuilt_structures.sha256)
    frozen = read_frozen_rebuilds(path)

    target_class = config.organometallic.target_class
    targets = [
        record
        for record in rejected
        if record.disposition == BROKEN and record.class_name == target_class
    ]
    target_indices = {record.index for record in targets}
    output.rejected = [
        record for record in rejected if record.index not in target_indices
    ]

    report = progress or (lambda _message: None)
    failed_validation = 0
    for record in targets:
        spec = parse_complex(record.smiles)
        if spec is None:
            output.rejected.append(_reject(record, "failed_parse"))
            continue
        frozen_rebuild = frozen.get(record.smiles)
        if frozen_rebuild is None:
            output.rejected.append(_reject(record, "no_frozen_geometry"))
            continue
        failure = validate_rebuild(spec, frozen_rebuild)
        if failure is not None:
            failed_validation += 1
            output.rejected.append(_reject(record, "failed_validation", failure))
            continue
        output.structures.append(
            Structure(
                index=record.index if record.index is not None else -1,
                smiles=spec.component_smiles,
                class_name=record.class_name,
                label=record.label,
                symbols=list(frozen_rebuild.symbols),
                coords=list(frozen_rebuild.coords),
                source_file=record.source_file,
                geometry_quality="rebuilt_architector",
                repair_strategy="architector",
                note=f"{spec.family}; rebuilt from fragmented source geometry",
            )
        )
    if failed_validation:
        report(f"  rebuild: {failed_validation} frozen geometries failed validation")

    output.counts = {
        "targets": len(targets),
        "rebuilt": len(output.structures),
        "failed": len(targets) - len(output.structures),
    }
    return output
