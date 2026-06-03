from collections.abc import Iterable

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

from threedscriptors.configuration.dataset_config import FilterMoleculeStageConfig
from threedscriptors.data_handling.dataset.tasks import ElementSet
from threedscriptors.data_handling.dataset_creation.build_stats import LoadStats
from threedscriptors.data_handling.dataset_creation.loading_batch import SmilesData

# MACE-OFF24 element coverage — drug-like organic subset.
MACE_OFF_ELEMENTS: set[str] = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}

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
MACE_POLAR_ELEMENTS: set[str] = set(_PERIODIC_TABLE_SYMBOLS)


def resolve_element_set(preset: ElementSet) -> set[str]:
    """Resolve an :class:`ElementSet` preset to the concrete element symbol set."""
    if preset is ElementSet.mace_off:
        return MACE_OFF_ELEMENTS
    if preset is ElementSet.mace_polar:
        return MACE_POLAR_ELEMENTS
    raise ValueError(f"Unknown ElementSet: {preset!r}")


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


def _classify_one_smiles(
    smi: str | None,
    *,
    max_atoms: int | None,
    allowed_elements: set[str],
    allow_charged: bool,
    allow_radicals: bool,
    allow_isotopes: bool,
    allow_multifragment: bool,
    strip_salts: bool,
    neutralize: bool,
) -> tuple[str, str | None]:
    """Parse → standardize → filter → canonicalize one SMILES.

    Returns ``(verdict, iso)`` where ``verdict`` is one of
    ``"invalid" | "filtered" | "kept"`` and ``iso`` is the canonical isomeric
    SMILES when the verdict is ``"kept"`` (else ``None``).
    """
    if smi is None:
        return "invalid", None
    mol = Chem.MolFromSmiles(smi)
    mol = standardize_mol(mol, strip_salts=strip_salts, neutralize=neutralize)
    if mol is None:
        return "invalid", None
    if not filter_mol(
        mol,
        max_atoms=max_atoms,
        allowed_elements=allowed_elements,
        allow_charged=allow_charged,
        allow_radicals=allow_radicals,
        allow_isotopes=allow_isotopes,
        allow_multifragment=allow_multifragment,
    ):
        return "filtered", None
    iso = Chem.MolToSmiles(
        Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True
    )
    return "kept", iso


def standardize_for_conformer(
    mol_implicit_h: Chem.Mol,
    cfg: FilterMoleculeStageConfig,
    *,
    seen: set[str] | None = None,
) -> str | None:
    """Same standardize → filter → canonicalize path as ``apply_smiles_filter``
    but tailored for 3D-source generators whose conformers must stay aligned
    with the standardized SMILES.

    Salt-stripping removes atoms; if that would happen we drop the molecule
    rather than emit a SMILES/conformer mismatch (the original ``rd_mol``
    still has the salt atoms). Pass ``seen`` to dedupe across calls.

    Returns the canonical isomeric SMILES, or ``None`` to drop the molecule.
    The input mol must be in implicit-H form (call ``Chem.RemoveAllHs`` first
    if the source supplied explicit hydrogens).
    """
    if mol_implicit_h is None:
        return None
    n_heavy = mol_implicit_h.GetNumHeavyAtoms()
    std = standardize_mol(
        mol_implicit_h, strip_salts=cfg.strip_salts, neutralize=cfg.neutralize
    )
    if std is None or std.GetNumHeavyAtoms() != n_heavy:
        return None
    allowed_elements = resolve_element_set(cfg.element_set)
    if not filter_mol(
        std,
        max_atoms=cfg.max_atoms,
        allowed_elements=allowed_elements,
        allow_charged=cfg.allow_charged,
        allow_radicals=cfg.allow_radicals,
        allow_isotopes=cfg.allow_isotopes,
        allow_multifragment=cfg.allow_multifragment,
    ):
        return None
    iso = Chem.MolToSmiles(
        Chem.RemoveAllHs(std), isomericSmiles=True, canonical=True
    )
    if cfg.dedupe and seen is not None:
        if iso in seen:
            return None
        seen.add(iso)
    return iso


def apply_smiles_filter(
    raw_smiles: Iterable[str | None],
    *,
    max_atoms: int | None,
    allowed_elements: set[str],
    allow_charged: bool = True,
    allow_radicals: bool = True,
    allow_isotopes: bool = False,
    allow_multifragment: bool = False,
    strip_salts: bool = False,
    neutralize: bool = False,
    dedupe: bool = True,
    seen: set[str] | None = None,
    stats: LoadStats | None = None,
) -> tuple[list[SmilesData], list[int]]:
    """Parse → standardize → filter → canonicalize → dedupe.

    Returns ``(kept_smiles_data, kept_indices)`` where ``kept_indices[k]`` is
    the row index into ``raw_smiles`` for ``kept_smiles_data[k]``. ``seen``
    is a caller-owned set used for dedupe; pass the same set across multiple
    calls to dedupe across batches with first-occurrence-wins semantics (this
    is how ``TdcGenerator`` keeps train > valid > test priority across the
    pyTDC frames). ``stats`` is an optional accumulator updated in place.
    """
    if seen is None:
        seen = set()
    local_stats = stats if stats is not None else LoadStats()
    kept_data: list[SmilesData] = []
    kept_indices: list[int] = []
    for i, smi in enumerate(raw_smiles):
        local_stats.n_raw_rows += 1
        verdict, iso = _classify_one_smiles(
            smi,
            max_atoms=max_atoms,
            allowed_elements=allowed_elements,
            allow_charged=allow_charged,
            allow_radicals=allow_radicals,
            allow_isotopes=allow_isotopes,
            allow_multifragment=allow_multifragment,
            strip_salts=strip_salts,
            neutralize=neutralize,
        )
        if verdict == "invalid":
            local_stats.n_invalid_smiles += 1
            continue
        if verdict == "filtered":
            local_stats.n_filtered_out += 1
            continue
        assert iso is not None
        if dedupe and iso in seen:
            local_stats.n_duplicates += 1
            continue
        if dedupe:
            seen.add(iso)
        kept_data.append(
            SmilesData(
                nonisomeric_smiles=Chem.CanonSmiles(iso, useChiral=False),
                isomeric_smiles=iso,
            )
        )
        kept_indices.append(i)
    local_stats.n_kept += len(kept_indices)
    return kept_data, kept_indices
