from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


# MACE-OFF24 element coverage — drug-like organic subset.
MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}

# MACE-POLAR-1-M element coverage — atomic numbers 1..83 (H through Bi).
# Verified 2026-05-03 by introspecting the model's atomic_numbers buffer.
# Covers all the metals + metalloids (Na, K, Ca, Mg, Li, Zn, Fe, B, Si, Se, Pt, Au, Hg, ...)
# that were causing pre-MACE molecule drops in salts / charged drug formulations.
_PERIODIC_TABLE_SYMBOLS = (
    "H He "
    "Li Be B C N O F Ne "
    "Na Mg Al Si P S Cl Ar "
    "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi"
).split()
MACE_POLAR_ELEMENTS = set(_PERIODIC_TABLE_SYMBOLS)


# Module-level singletons for repeated standardization calls.
_SALT_REMOVER = SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_mol(
    mol: Chem.Mol,
    *,
    strip_salts: bool = False,
    neutralize: bool = False,
) -> Chem.Mol | None:
    """Standardize an RDKit Mol with optional salt-stripping + uncharging.

    Both flags default to False so existing call sites are unaffected. Pass
    ``strip_salts=True, neutralize=True`` for the POLAR build path.

    Returns the standardized Mol, or ``None`` if standardization removed all
    atoms (e.g. molecule was a pure counter-ion).
    """
    if mol is None:
        return None
    try:
        if strip_salts:
            mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
            if mol is None or mol.GetNumAtoms() == 0:
                return None
        if neutralize:
            mol = _UNCHARGER.uncharge(mol)
            if mol is None:
                return None
        return mol
    except Exception:
        return None


def filter_mol(
    mol: Chem.Mol,
    require_3D: bool = False,
    max_atoms: int | None = None,
    *,
    allowed_elements: set[str] | None = None,
    allow_charged: bool = True,
    allow_radicals: bool = True,
    allow_isotopes: bool = False,
    allow_multifragment: bool = False,
) -> bool:
    """Return True if mol passes all filters, otherwise False.

    The legacy signature ``filter_mol(mol, require_3D, max_atoms)`` is preserved
    bit-for-bit: with the default keyword args this behaves exactly like the
    original (reject <3 atoms, >max_atoms, multi-fragment, isotopes, and any
    element outside ``MACE_OFF_ELEMENTS``; charge and radicals are NOT
    rejected). Every existing generator caller is therefore unaffected.

    ``allow_charged`` / ``allow_radicals`` default to True *specifically* to
    preserve that historical behaviour for the existing dataset generators.
    EVAL-001 opts into the stricter MACE-OFF24 / MACE-POLAR semantics
    explicitly via ``allowed_elements=..., allow_charged=False,
    allow_radicals=False``.
    """
    if allowed_elements is None:
        allowed_elements = MACE_OFF_ELEMENTS
    try:
        if mol is None:
            return False
        # must have a 3D conformer
        if require_3D and (mol.GetNumConformers() == 0):
            return False
        # atom-count bounds
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 3:
            return False
        if max_atoms is not None and num_atoms > max_atoms:
            return False
        # single fragment only (unless caller opts in to multi-frag)
        if not allow_multifragment:
            if (
                Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
                and len(Chem.GetMolFrags(mol, asMols=True)) > 1
            ):
                return False

        if not allow_charged and Chem.GetFormalCharge(mol) != 0:
            return False

        for a in mol.GetAtoms():
            if a.GetSymbol() not in allowed_elements:
                return False
            if not allow_radicals and a.GetNumRadicalElectrons() != 0:
                return False
            if not allow_isotopes and a.GetIsotope() != 0:
                return False
        return True
    except Exception:
        return False
