"""Shared RDKit physicochemical descriptor registry.

Single source of truth for the cheap RDKit descriptors used both by the dataset
analysis path and by the dataset-creation ``PhysicochemicalDescriptorStage``
(probe targets).

Two registries:

* ``DESCRIPTORS_2D`` — topological descriptors evaluated on a SMILES-derived
  ``Chem.Mol`` (no geometry needed).
* ``DESCRIPTORS_3D`` — geometry-dependent descriptors (solvent accessible surface
  area) evaluated on a ``Chem.Mol`` that carries a 3D conformer + perceived
  connectivity.

The split makes the descriptors *provenance-independent*: 2D descriptors come
from ``MolFromSmiles(iso)`` and 3D descriptors from a mol built directly out of
atomic numbers + positions, so nothing relies on a SMILES<->geometry atom-order
correspondence (which does NOT hold for SDF-loaded structures, whose atom order
follows the SDF mol rather than ``AddHs(MolFromSmiles)``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from ase.data import chemical_symbols
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdDetermineBonds, rdFreeSASA
from rdkit.Chem import rdMolDescriptors as rdMD

# RDKit is chatty about sanitization / valence in worker processes.
RDLogger.DisableLog("rdApp.*")


def _sasa(mol: Chem.Mol) -> float:
    """Total solvent accessible surface area (Å²) for a 3D mol."""
    radii = rdFreeSASA.classifyAtoms(mol)
    return float(rdFreeSASA.CalcSASA(mol, radii))


# Polar atoms for the polar-SASA fraction. Element-set-agnostic so it scales to
# the broad element coverage of large datasets (mace_polar spans Z=1..83):
# carbon and carbon-bound hydrogens are apolar, every other element (every
# heteroatom, plus hydrogens bound to one) is treated as polar. This avoids
# enumerating heteroatoms and does not depend on RDKit's Protor classifier
# (which returns "Unclassified" for plain mols built from xyz).
def _is_polar_atom(atom: Chem.Atom) -> bool:
    z = atom.GetAtomicNum()
    if z == 6:  # carbon: apolar
        return False
    if z == 1:  # hydrogen: polar iff bonded to a non-carbon heavy atom
        return any(nbr.GetAtomicNum() != 6 for nbr in atom.GetNeighbors())
    return True  # any other element is a (polar) heteroatom


def _polar_sasa_fraction(mol: Chem.Mol) -> float:
    """Fraction of SASA contributed by polar atoms (NaN if total SASA is zero).

    ``CalcSASA`` writes a per-atom ``SASA`` property; the polar contribution is
    the sum over N/O atoms and their attached hydrogens.
    """
    radii = rdFreeSASA.classifyAtoms(mol)
    total = rdFreeSASA.CalcSASA(mol, radii)
    if total <= 0.0:
        return float("nan")
    polar = sum(
        atom.GetDoubleProp("SASA") for atom in mol.GetAtoms() if _is_polar_atom(atom)
    )
    return float(polar / total)


DESCRIPTORS_2D: dict[str, Callable[[Chem.Mol], float]] = {
    "mw": lambda m: float(Descriptors.MolWt(m)),
    "n_heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "tpsa": lambda m: float(rdMD.CalcTPSA(m)),
    "hbd": lambda m: float(rdMD.CalcNumHBD(m)),
    "hba": lambda m: float(rdMD.CalcNumHBA(m)),
    "rot_bonds": lambda m: float(rdMD.CalcNumRotatableBonds(m)),
    "fraction_csp3": lambda m: float(rdMD.CalcFractionCSP3(m)),
    "n_rings": lambda m: float(rdMD.CalcNumRings(m)),
    "n_aromatic_rings": lambda m: float(rdMD.CalcNumAromaticRings(m)),
    "qed": lambda m: float(QED.qed(m)),
    "mol_mr": lambda m: float(Crippen.MolMR(m)),
}

DESCRIPTORS_3D: dict[str, Callable[[Chem.Mol], float]] = {
    "sasa": _sasa,
    "polar_sasa_fraction": _polar_sasa_fraction,
}

# Default probe target columns (config-overridable). ``mw`` + ``n_heavy_atoms``
# are the only size positive-controls; the rest target chemistry beyond raw atom
# count.
DEFAULT_DESCRIPTORS: tuple[str, ...] = (
    "mw",
    "n_heavy_atoms",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rot_bonds",
    "fraction_csp3",
    "n_aromatic_rings",
    "qed",
    "sasa",
    "polar_sasa_fraction",
)


def split_names(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Partition descriptor names into (2D, 3D); raise on unknown names."""
    names_2d = [n for n in names if n in DESCRIPTORS_2D]
    names_3d = [n for n in names if n in DESCRIPTORS_3D]
    unknown = [n for n in names if n not in DESCRIPTORS_2D and n not in DESCRIPTORS_3D]
    if unknown:
        raise KeyError(
            f"Unknown physchem descriptors: {unknown}. "
            f"Known 2D={sorted(DESCRIPTORS_2D)} 3D={sorted(DESCRIPTORS_3D)}"
        )
    return names_2d, names_3d


def compute_2d(iso_smiles: str | None, names_2d: Sequence[str]) -> tuple[np.ndarray, bool]:
    """2D descriptors from a SMILES. Returns (values, ok); ok=False => all-NaN."""
    out = np.full(len(names_2d), np.nan, dtype=np.float64)
    if not iso_smiles:
        return out, False
    mol = Chem.MolFromSmiles(iso_smiles)
    if mol is None:
        return out, False
    for i, name in enumerate(names_2d):
        try:
            out[i] = DESCRIPTORS_2D[name](mol)
        except Exception:
            out[i] = np.nan
    return out, True


def _to_xyz_block(atomic_numbers: Sequence[int], positions: np.ndarray) -> str:
    pos = np.asarray(positions, dtype=np.float64)
    lines = [str(len(atomic_numbers)), ""]
    for z, (x, y, zc) in zip(atomic_numbers, pos, strict=True):
        lines.append(f"{chemical_symbols[int(z)]} {x:.6f} {y:.6f} {zc:.6f}")
    return "\n".join(lines) + "\n"


def mol_from_atoms(
    atomic_numbers: Sequence[int], positions: np.ndarray
) -> Chem.Mol | None:
    """Build a 3D RDKit mol with perceived connectivity from xyz.

    Connectivity is perceived geometrically (charge-free) so it works for loaded
    structures regardless of provenance. Returns ``None`` on any failure (caller
    masks the 3D columns).
    """
    try:
        mol = Chem.MolFromXYZBlock(_to_xyz_block(atomic_numbers, positions))
        if mol is None:
            return None
        rdDetermineBonds.DetermineConnectivity(mol)
        return mol
    except Exception:
        return None


def compute_3d(mol3d: Chem.Mol | None, names_3d: Sequence[str]) -> tuple[np.ndarray, bool]:
    """3D descriptors from a geometry mol. Returns (values, ok)."""
    out = np.full(len(names_3d), np.nan, dtype=np.float64)
    if mol3d is None:
        return out, False
    for i, name in enumerate(names_3d):
        try:
            out[i] = DESCRIPTORS_3D[name](mol3d)
        except Exception:
            out[i] = np.nan
    return out, True


def assemble_row(
    names: Sequence[str],
    names_2d: Sequence[str],
    values_2d: np.ndarray,
    names_3d: Sequence[str],
    values_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge 2D/3D values back into ``names`` order, building a finite-value mask."""
    val2 = dict(zip(names_2d, values_2d, strict=True))
    val3 = dict(zip(names_3d, values_3d, strict=True))
    values = np.empty(len(names), dtype=np.float64)
    mask = np.zeros(len(names), dtype=np.uint8)
    for i, name in enumerate(names):
        v = val2[name] if name in val2 else val3[name]
        values[i] = v
        mask[i] = 1 if np.isfinite(v) else 0
    return values, mask


def compute_row(
    names: Sequence[str],
    iso_smiles: str | None,
    atomic_numbers: Sequence[int],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: full descriptor row + finite-mask for one structure.

    Column order follows ``names``. 2D columns use the SMILES; 3D columns use the
    geometry. Either source failing leaves its columns NaN/masked.
    """
    names_2d, names_3d = split_names(names)
    v2, _ = compute_2d(iso_smiles, names_2d)
    mol3d = mol_from_atoms(atomic_numbers, positions) if names_3d else None
    v3, _ = compute_3d(mol3d, names_3d)
    return assemble_row(names, names_2d, v2, names_3d, v3)
