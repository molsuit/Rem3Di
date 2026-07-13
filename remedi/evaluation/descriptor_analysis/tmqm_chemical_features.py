"""Per-structure chemical descriptors for tmQM transition-metal complexes.

These features are *interpretation aids*: they let us ask whether an
unsupervised descriptor cluster shares a chemical motif (oxygen-donor,
phosphine, sandwich / arene, diimine octahedral, carborane, ...). They are
derived purely from geometry + composition, so they work on the same data the
descriptors see (no SMILES / bond orders required, none are stored for tmQM).

The metal first-coordination sphere is inferred with the same
``MinimumDistanceNN`` heuristic already used for the coordination-number color
provider, so values stay consistent across the analysis suite.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from pydantic import BaseModel
from pymatgen.core import Molecule
from pymatgen.core.local_env import MinimumDistanceNN

from remedi.evaluation.descriptor_analysis.tmqm_clustering_utils import (
    TM_numbers,
)

HALOGENS = {"F", "Cl", "Br", "I", "At"}

# Largest C-C separation (Angstrom) still treated as a bond when deciding
# whether several carbon donors form one contiguous pi system (Cp ~1.42,
# arene ~1.40, slightly stretched on coordination -> 1.8 is comfortably safe).
_CC_BOND_MAX = 1.8

# Generic heavy-heavy single-bond cutoff (Angstrom). Used to build the backbone
# graph for chelate-ring detection; loose enough to catch coordinated ligands
# (M-N, M-O, etc. are excluded explicitly via metal_idx).
_HEAVY_BOND_MAX = 1.85

# Carbonyl C=O / C#O bond length is ~1.13-1.18; allow a margin.
_CO_BOND_MAX = 1.30

# Chelate ring sizes considered "real" (excluding the trivial monodentate case).
_CHELATE_RING_MIN = 4
_CHELATE_RING_MAX = 6

# n_B at/above this is taken as a carborane / borane-cage signature rather than
# an incidental boryl/borane ligand.
_CARBORANE_MIN_B = 4


def metal_block(z: int) -> str:
    """Map an atomic number to its d/f sub-shell row label."""
    if 21 <= z <= 30:
        return "3d"
    if 39 <= z <= 48:
        return "4d"
    if z == 57 or 72 <= z <= 80:
        return "5d"
    if 58 <= z <= 71:
        return "4f"
    if 89 <= z <= 103:
        return "5f"
    return "other"


class MetalEnvironmentFeatures(BaseModel):
    """Geometry/composition descriptors for one TM complex.

    ``coordination_number == -1`` and ``donor_set == ""`` flag a structure
    where the metal environment could not be resolved (no unique TM center or
    a pymatgen failure); whole-molecule composition fields are still valid.
    """

    metal_symbol: str
    metal_block: str
    coordination_number: int
    donor_set: str
    donor_counts: dict[str, int]
    n_donor_C: int
    n_donor_N: int
    n_donor_O: int
    n_donor_P: int
    n_donor_S: int
    n_donor_halogen: int
    n_donor_H: int = 0
    """Hydride donors (M-H), directly counted from the first sphere."""
    n_carbonyl: int = 0
    """Terminal C≡O ligands: C donors bonded to exactly one O at < 1.30 Å."""
    hapticity_max: int
    n_pi_groups: int
    n_chelate_rings: int = 0
    """Donor pairs connected by a 2-4-bond backbone (chelate ring size 4-6)."""
    max_chelate_ring_size: int = 0
    chelate_ring_sizes: list[int] = []
    geometry_class: str = "unknown"
    """Rough coordination geometry inferred from donor-M-donor angles."""
    radius_of_gyration: float = 0.0
    n_atoms: int
    formula: str
    n_B: int
    n_C: int
    n_N: int
    n_O: int
    n_P: int
    n_S: int
    n_halogen: int
    is_sandwich: bool
    is_carborane: bool


def _find_metal_index(numbers: np.ndarray) -> int | None:
    metal_positions = [i for i, z in enumerate(numbers) if int(z) in TM_numbers]
    if len(metal_positions) != 1:
        return None
    return metal_positions[0]


def _canonical_donor_set(symbols: list[str]) -> tuple[str, dict[str, int]]:
    counts = Counter(symbols)
    # Carbon first (pi/organometallic motifs), then alphabetical: gives stable
    # human-readable signatures like "N4O2", "C5C5"->"C10", "P2Cl2".
    ordered = sorted(counts.items(), key=lambda kv: (kv[0] != "C", kv[0]))
    label = "".join(f"{el}{n}" for el, n in ordered)
    return label, dict(ordered)


def _carbon_pi_groups(
    coords: np.ndarray, carbon_local_idx: list[int]
) -> tuple[int, int]:
    """Return (largest contiguous carbon-donor group, #groups of size>=3).

    Two coordinating carbons are linked when within ``_CC_BOND_MAX``; a
    connected component is one pi ligand. Cp -> 5, arene -> 6, ferrocene-like
    sandwich -> two groups.
    """
    n = len(carbon_local_idx)
    if n == 0:
        return 0, 0
    pts = coords[carbon_local_idx]
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) <= _CC_BOND_MAX:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    sizes: list[int] = []
    for start in range(n):
        if seen[start]:
            continue
        stack, comp = [start], 0
        seen[start] = True
        while stack:
            node = stack.pop()
            comp += 1
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        sizes.append(comp)

    return max(sizes), sum(1 for s in sizes if s >= 3)


def _count_carbonyls(
    positions: np.ndarray,
    symbols: list[str],
    metal_idx: int,
    carbon_donor_idx: list[int],
) -> int:
    """Count terminal CO ligands among the carbon donors.

    A carbonyl carbon is bonded to the metal *and* to exactly one O at the
    short C=O/C≡O distance, with no other heavy neighbour (besides the metal).
    """
    n_co = 0
    for c in carbon_donor_idx:
        pos_c = positions[c]
        co_o = 0
        other_heavy = 0
        for i, sym in enumerate(symbols):
            if i in (c, metal_idx) or sym == "H":
                continue
            d = float(np.linalg.norm(positions[i] - pos_c))
            if sym == "O" and d <= _CO_BOND_MAX:
                co_o += 1
            elif d <= _HEAVY_BOND_MAX:
                other_heavy += 1
        if co_o == 1 and other_heavy == 0:
            n_co += 1
    return n_co


def _heavy_atom_adjacency(
    positions: np.ndarray, symbols: list[str], metal_idx: int
) -> dict[int, list[int]]:
    """Bond graph over heavy atoms excluding the metal (distance < ~1.85 Å)."""
    heavy = [i for i in range(len(symbols)) if i != metal_idx and symbols[i] != "H"]
    adj: dict[int, list[int]] = {i: [] for i in heavy}
    for ai, a in enumerate(heavy):
        for b in heavy[ai + 1 :]:
            if float(np.linalg.norm(positions[a] - positions[b])) <= _HEAVY_BOND_MAX:
                adj[a].append(b)
                adj[b].append(a)
    return adj


def _shortest_backbone_distance(
    adj: dict[int, list[int]],
    start: int,
    goal: int,
    max_len: int,
    blocked: set[int],
) -> int | None:
    """BFS shortest-path length in ``adj`` from ``start`` to ``goal``.

    Bounded by ``max_len``; cannot route through any node in ``blocked``
    (typically other donor atoms — keeps each donor-pair ring primitive).
    """
    if start == goal:
        return 0
    visited = {start: 0}
    frontier = [start]
    while frontier:
        next_frontier: list[int] = []
        for u in frontier:
            if visited[u] >= max_len:
                continue
            for v in adj[u]:
                if v in visited:
                    continue
                visited[v] = visited[u] + 1
                if v == goal:
                    return visited[v]
                if v in blocked:
                    continue
                next_frontier.append(v)
        frontier = next_frontier
    return None


def _chelate_ring_sizes(
    positions: np.ndarray,
    symbols: list[str],
    donor_local: list[int],
    metal_idx: int,
) -> list[int]:
    """Per-donor-pair chelate ring size (atoms incl. metal), if 4-6 membered.

    Pairs of donors connected by a 2-4-bond backbone (excluding the metal)
    form a 4-6 chelate ring; monodentate donors are simply skipped.
    """
    adj = _heavy_atom_adjacency(positions, symbols, metal_idx)
    donors = [d for d in donor_local if d in adj]
    max_path = _CHELATE_RING_MAX - 2
    donor_set = set(donors)
    rings: list[int] = []
    for i in range(len(donors)):
        for j in range(i + 1, len(donors)):
            a, b = donors[i], donors[j]
            path = _shortest_backbone_distance(adj, a, b, max_path, donor_set - {a, b})
            if path is None:
                continue
            size = path + 2
            if _CHELATE_RING_MIN <= size <= _CHELATE_RING_MAX:
                rings.append(size)
    return rings


def _classify_geometry(
    positions: np.ndarray, metal_idx: int, donor_local: list[int], cn: int
) -> str:
    """Coarse coordination-geometry label from donor-M-donor angles.

    Strict enough to recover textbook classes (tetrahedral / square-planar /
    octahedral / TBP / SP) and degrade to "distorted_*" / "high_cn_*" rather
    than mis-label outliers.
    """
    if cn <= 0:
        return "unknown"
    if cn == 1:
        return "monocoordinate"

    m = positions[metal_idx]
    vecs = np.asarray([positions[i] - m for i in donor_local])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1.0)
    cos = np.clip(vecs @ vecs.T, -1.0, 1.0)
    iu = np.triu_indices(cn, k=1)
    angles = np.degrees(np.arccos(cos[iu]))

    if cn == 2:
        return "linear" if angles[0] > 160 else "bent"
    if cn == 3:
        return "trigonal_planar" if 110 < float(angles.mean()) < 130 else "T-shape"
    if cn == 4:
        return "tetrahedral" if float(angles.mean()) > 100 else "square_planar"
    if cn == 5:
        beta, alpha = float(np.sort(angles)[-1]), float(np.sort(angles)[-2])
        return (
            "trigonal_bipyramidal" if (beta - alpha) / 60 > 0.5 else "square_pyramidal"
        )
    if cn == 6:
        n_cis = int(np.sum((angles > 75) & (angles < 105)))
        n_trans = int(np.sum(angles > 160))
        return "octahedral_like" if (n_cis >= 10 and n_trans >= 2) else "distorted_6"
    return f"high_cn_{cn}"


def _radius_of_gyration(positions: np.ndarray) -> float:
    com = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1))))


def _features_for_atoms(atoms: Atoms) -> MetalEnvironmentFeatures:
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    symbols = [chemical_symbols[int(z)] for z in numbers]
    elem_counts = Counter(symbols)

    n_halogen = sum(elem_counts[h] for h in HALOGENS)
    common = {
        "n_B": elem_counts["B"],
        "n_C": elem_counts["C"],
        "n_N": elem_counts["N"],
        "n_O": elem_counts["O"],
        "n_P": elem_counts["P"],
        "n_S": elem_counts["S"],
        "n_halogen": n_halogen,
    }

    metal_idx = _find_metal_index(numbers)
    metal_symbol = "?"
    block = "other"
    if metal_idx is not None:
        z = int(numbers[metal_idx])
        metal_symbol = chemical_symbols[z]
        block = metal_block(z)

    cn = -1
    donor_set = ""
    donor_counts: dict[str, int] = {}
    n_donor = dict.fromkeys("CNOPS", 0)
    n_donor_halogen = 0
    n_donor_H = 0
    n_carbonyl = 0
    hapticity_max = 0
    n_pi_groups = 0
    chelate_rings: list[int] = []
    geometry = "unknown"

    positions = atoms.get_positions()
    if metal_idx is not None:
        try:
            mol = Molecule(
                species=symbols,
                coords=positions,
                charge=round(float(atoms.info.get("total_charge", 0.0))),
                spin_multiplicity=None,
            )
            nn = MinimumDistanceNN(tol=0.20)
            nn_info = nn.get_nn_info(
                mol, metal_idx
            )  # ty: ignore[invalid-argument-type]
            donor_local = [int(nn["site_index"]) for nn in nn_info]
            donor_symbols = [symbols[i] for i in donor_local]
            cn = len(donor_local)
            donor_set, donor_counts = _canonical_donor_set(donor_symbols)
            for s in donor_symbols:
                if s in n_donor:
                    n_donor[s] += 1
            n_donor_halogen = sum(1 for s in donor_symbols if s in HALOGENS)
            n_donor_H = sum(1 for s in donor_symbols if s == "H")
            carbon_local = [i for i in donor_local if symbols[i] == "C"]
            hapticity_max, n_pi_groups = _carbon_pi_groups(positions, carbon_local)
            n_carbonyl = _count_carbonyls(positions, symbols, metal_idx, carbon_local)
            chelate_rings = _chelate_ring_sizes(
                positions, symbols, donor_local, metal_idx
            )
            geometry = _classify_geometry(positions, metal_idx, donor_local, cn)
        except Exception:
            # Geometry too pathological for the NN heuristic; keep composition.
            cn = -1

    return MetalEnvironmentFeatures(
        metal_symbol=metal_symbol,
        metal_block=block,
        coordination_number=cn,
        donor_set=donor_set,
        donor_counts=donor_counts,
        n_donor_C=n_donor["C"],
        n_donor_N=n_donor["N"],
        n_donor_O=n_donor["O"],
        n_donor_P=n_donor["P"],
        n_donor_S=n_donor["S"],
        n_donor_halogen=n_donor_halogen,
        n_donor_H=n_donor_H,
        n_carbonyl=n_carbonyl,
        hapticity_max=hapticity_max,
        n_pi_groups=n_pi_groups,
        n_chelate_rings=len(chelate_rings),
        max_chelate_ring_size=max(chelate_rings) if chelate_rings else 0,
        chelate_ring_sizes=sorted(chelate_rings),
        geometry_class=geometry,
        radius_of_gyration=_radius_of_gyration(positions),
        n_atoms=len(numbers),
        formula=atoms.get_chemical_formula(mode="hill"),
        is_sandwich=n_pi_groups >= 2 and hapticity_max >= 5,
        is_carborane=common["n_B"] >= _CARBORANE_MIN_B,
        **common,
    )


def compute_metal_environment_features(
    molecules: list[Atoms],
) -> list[MetalEnvironmentFeatures]:
    """Vectorize :func:`_features_for_atoms` over a dataset of complexes."""
    return [_features_for_atoms(m) for m in molecules]
