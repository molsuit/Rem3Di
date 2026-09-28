"""Shared RDKit helpers: standardization, hydrogen handling, clash detection.

Every stage that touches a molecule goes through this module, so the geometry
produced by the extraction and the geometry checked by the repair stage are
built the same way.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Geometry import Point3D
from scipy.spatial.distance import pdist, squareform

from .records import Coordinate

# Module-level singletons for repeated standardization calls. Verified
# stereo-safe: Uncharger only edits formal charge / hydrogen counts, and
# SaltRemover only removes whole counter-ion fragments.
_SALT_REMOVER = SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()

# Approximate X-H bond lengths (Angstrom) for repairing misplaced hydrogens.
_HYDROGEN_BOND_LENGTH: dict[int, float] = {
    6: 1.09,
    7: 1.01,
    8: 0.97,
    9: 0.92,
    15: 1.42,
    16: 1.34,
    17: 1.27,
    35: 1.41,
    53: 1.61,
}

# Bare metal atoms that mark a true organometallic (out of ordinary repair scope).
METAL_SYMBOLS: frozenset[str] = frozenset(
    [
        "Fe",
        "Ru",
        "Rh",
        "Ir",
        "Pd",
        "Pt",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Mn",
        "Cr",
        "Mo",
        "W",
        "V",
        "Ti",
        "Os",
        "Re",
        "Ag",
        "Au",
        "Hg",
        "Sn",
        "Pb",
        "Mg",
        "Ca",
        "Al",
    ]
)

# Chirality that the canonical isomeric SMILES fully encodes -> safe to re-embed.
SMILES_ENCODED_CLASSES: frozenset[str] = frozenset({"central", "achiral"})


# --------------------------------------------------------------------------- #
# SMILES / molecule helpers
# --------------------------------------------------------------------------- #


def canonical_smiles(smiles: str) -> str | None:
    """Canonical isomeric SMILES, or ``None`` if the string does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_mols(raw_mol: object) -> list[Chem.Mol]:
    """Return RDKit mols from either a CommonChem JSON string or a list."""
    if isinstance(raw_mol, str):
        return list(Chem.JSONToMols(raw_mol))
    if isinstance(raw_mol, (list, tuple)):
        return list(raw_mol)
    raise TypeError(f"Unsupported `mol` payload of type {type(raw_mol).__name__}")


def pick_conformer_id(mol: Chem.Mol, conformer_id: int) -> int:
    conformer_ids = [conformer.GetId() for conformer in mol.GetConformers()]
    return conformer_id if conformer_id in conformer_ids else conformer_ids[0]


def conformer_coords(mol: Chem.Mol, conformer_id: int) -> list[Coordinate]:
    conformer = mol.GetConformer(conformer_id)
    return [tuple(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]


def mol_to_arrays(mol: Chem.Mol) -> tuple[list[str], list[Coordinate]]:
    """Split a single-conformer mol into (symbols, coordinates)."""
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = [(float(x), float(y), float(z)) for x, y, z in coords_array(mol)]
    return symbols, coords


def coords_array(mol: Chem.Mol) -> np.ndarray:
    conformer = mol.GetConformer()
    return np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )


def single_conformer(mol: Chem.Mol, conformer_id: int) -> Chem.Mol:
    """Sanitised copy of ``mol`` carrying only the chosen 3D conformer.

    ``JSONToMols`` returns unsanitised mols; sanitisation (perceived bonds and
    hybridisation) is needed before AddHs, SaltRemover and Uncharger behave
    correctly. Sanitisation failures (organometallics) fall through unsanitised.
    """
    base = Chem.Mol(mol)
    base.RemoveAllConformers()
    base.AddConformer(mol.GetConformer(conformer_id), assignId=True)
    base.GetConformer(0).Set3D(True)
    with contextlib.suppress(Exception):
        Chem.SanitizeMol(base)
    return base


def standardize_mol(
    mol: Chem.Mol, *, strip_salts: bool, neutralize: bool
) -> Chem.Mol | None:
    """Strip counter-ion salts and/or neutralize charges, preserving the conformer.

    Returns ``None`` only if salt stripping removed every atom. Each step is
    guarded so an exotic structure (e.g. an organometallic) is kept unchanged
    rather than dropped.
    """
    if strip_salts:
        try:
            stripped = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
            if stripped is None or stripped.GetNumAtoms() == 0:
                return None
            mol = stripped
        except Exception:
            pass
    if neutralize:
        with contextlib.suppress(Exception):
            mol = _UNCHARGER.uncharge(mol)
    return mol


def repair_hydrogen_coords(mol: Chem.Mol) -> int:
    """Reposition any hydrogen whose bond to its heavy neighbour is implausible.

    ``AddHs(addCoords=True)`` occasionally drops a hydrogen at the origin. Such
    hydrogens are placed deterministically opposite the sum of the heavy atom's
    other neighbour directions (a standard idealised-geometry heuristic).
    Returns the number of repaired hydrogens.
    """
    conformer = mol.GetConformer(0)

    def position(index: int) -> np.ndarray:
        return np.array(conformer.GetAtomPosition(index))

    heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]

    repaired = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        heavy = neighbors[0] if neighbors else None
        if (
            heavy is not None
            and 0.7
            <= float(np.linalg.norm(position(atom.GetIdx()) - position(heavy.GetIdx())))
            <= 1.4
        ):
            continue  # already well placed

        # Choose a reference heavy atom: the bonded neighbour, else the nearest
        # heavy atom in 3D (handles hydrogens orphaned by failed sanitisation).
        if heavy is not None:
            reference = heavy.GetIdx()
            sibling_indices = [n.GetIdx() for n in heavy.GetNeighbors()]
        elif heavy_indices:
            hydrogen_position = position(atom.GetIdx())
            reference = min(
                heavy_indices,
                key=lambda j: float(np.linalg.norm(position(j) - hydrogen_position)),
            )
            sibling_indices = []
        else:
            continue  # nothing to attach to

        anchor = position(reference)
        directions = []
        for sibling in sibling_indices:
            if sibling == atom.GetIdx():
                continue
            offset = position(sibling) - anchor
            norm = float(np.linalg.norm(offset))
            if norm > 1e-3:
                directions.append(offset / norm)
        if directions:
            summed = np.sum(directions, axis=0)
            summed_norm = float(np.linalg.norm(summed))
            direction = (
                -summed / summed_norm if summed_norm > 1e-3 else np.array([0.0, 0.0, 1.0])
            )
        else:
            direction = np.array([0.0, 0.0, 1.0])
        bond_length = _HYDROGEN_BOND_LENGTH.get(
            mol.GetAtomWithIdx(reference).GetAtomicNum(), 1.09
        )
        placed = anchor + bond_length * direction
        conformer.SetAtomPosition(
            atom.GetIdx(),
            Point3D(float(placed[0]), float(placed[1]), float(placed[2])),
        )
        repaired += 1
    return repaired


def add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Add explicit, 3D-placed hydrogens to a sanitised single-conformer mol."""
    mol_with_hydrogens = Chem.AddHs(mol, addCoords=True)
    repair_hydrogen_coords(mol_with_hydrogens)
    return mol_with_hydrogens


# --------------------------------------------------------------------------- #
# Clash detection (element-aware bond-length floors)
# --------------------------------------------------------------------------- #


def pair_floor(symbol_a: str, symbol_b: str) -> float:
    """Minimum plausible distance (Angstrom) for an atom pair of these elements."""
    if symbol_a == "H" and symbol_b == "H":
        return 0.7
    if symbol_a == "H" or symbol_b == "H":
        return 0.74
    return 0.9


@dataclass
class Clash:
    atom_a: int
    symbol_a: str
    atom_b: int
    symbol_b: str
    distance: float


def find_clashes(symbols: list[str], coords: np.ndarray) -> list[Clash]:
    """Return every atom pair closer than its element-aware floor, nearest first."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return []
    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)
    clashes: list[Clash] = []
    rows, cols = np.triu_indices(len(coords), k=1)
    for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
        distance = float(distances[a, b])
        if distance < pair_floor(symbols[a], symbols[b]):
            clashes.append(Clash(a, symbols[a], b, symbols[b], round(distance, 3)))
    clashes.sort(key=lambda clash: clash.distance)
    return clashes


def categorize_clash(symbols: list[str], smiles: str, clashes: list[Clash]) -> str:
    """Map a clashing structure to a repair category."""
    if any(clash.distance < 0.1 for clash in clashes):
        return "exact_dup_organometallic"
    if "." in smiles:
        if any(symbol in METAL_SYMBOLS for symbol in symbols):
            return "metal_organometallic"
        return "multi_fragment_salt"
    if any(symbol in METAL_SYMBOLS for symbol in symbols):
        return "metal_organometallic"
    if all(clash.symbol_a == "H" or clash.symbol_b == "H" for clash in clashes):
        return "hydrogen_artifact"
    return "collapsed_geometry_organic"


# --------------------------------------------------------------------------- #
# Stereochemistry
# --------------------------------------------------------------------------- #


def stereo_signature_from_3d(mol: Chem.Mol) -> str | None:
    """Canonical isomeric SMILES with stereo re-perceived from the 3D coords.

    This is the ground-truth chirality of a geometry: it ignores whatever stereo
    flags the mol carries and reads R/S (and E/Z) straight from the conformer.
    Returns ``None`` if the heavy-atom skeleton cannot be processed.
    """
    try:
        working = Chem.RemoveHs(Chem.Mol(mol))
        Chem.AssignStereochemistryFrom3D(working)
        return Chem.MolToSmiles(working)
    except Exception:
        return None


def chirality_preserved(intended_smiles: str, repaired: Chem.Mol) -> tuple[bool, str]:
    """True iff the repaired geometry's 3D-perceived stereo matches the label."""
    intended = canonical_smiles(intended_smiles)
    achieved = stereo_signature_from_3d(repaired)
    if intended is None or achieved is None:
        return False, achieved or ""
    return achieved == intended, achieved


def identity_signature(mol: Chem.Mol) -> str | None:
    """Canonical isomeric SMILES (heavy atoms) carrying the mol's stereo flags."""
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    except Exception:
        return None


def has_unspecified_stereo(smiles: str) -> bool:
    """True if the SMILES leaves a real stereo element unassigned.

    Such a molecule cannot be safely re-embedded: ETKDG would pick an arbitrary
    enantiomer, so there is no defined chirality to preserve.
    """
    from rdkit.Chem import FindPotentialStereo

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        info = FindPotentialStereo(mol)
    except Exception:
        return False
    return any(str(element.specified) == "Unspecified" for element in info)


def count_stereocenters_from_3d(mol_with_hydrogens: Chem.Mol) -> tuple[int, int]:
    """Return (assigned R/S centres, tetrahedral-tagged atoms) read from 3D."""
    Chem.AssignStereochemistryFrom3D(mol_with_hydrogens)
    assigned = Chem.FindMolChiralCenters(
        mol_with_hydrogens,
        force=True,
        includeUnassigned=False,
        useLegacyImplementation=False,
    )
    tagged = sum(
        1
        for atom in mol_with_hydrogens.GetAtoms()
        if atom.GetChiralTag()
        in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    )
    return len(assigned), tagged
