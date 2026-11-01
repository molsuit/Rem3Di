"""The SMILES-side cleaning every SMILES-bearing source goes through.

Parse -> standardize (strip salts, neutralize) -> filter (size, elements,
charge, radicals, isotopes, fragments) -> canonicalize -> optionally
deduplicate. One settings model, :class:`SmilesFilterConfig`, drives every
caller: the preparers, the pipeline's ``FilterMoleculeStage`` and the 3D-source
generators.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

from remedi.data_handling.chemistry.elements import (
    ElementSet,
    allowed_element_symbols,
)


class SmilesFilterConfig(BaseModel):
    """The knobs of the SMILES filter.

    ``element_set`` is a named preset or an explicit list of element symbols;
    ``None`` switches the element gate off. ``max_atoms`` counts the atoms of
    the implicit-hydrogen molecule, i.e. heavy atoms for a parsed SMILES.
    """

    model_config = ConfigDict(extra="forbid")

    max_atoms: int | None = 100
    element_set: ElementSet | list[str] | None = ElementSet.mace_off

    allow_charged: bool = True
    allow_radicals: bool = True
    allow_isotopes: bool = False
    allow_multifragment: bool = False

    strip_salts: bool = True
    neutralize: bool = True

    dedupe: bool = True


# Module-level singletons for repeated standardization calls.
_SALT_REMOVER = SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_mol(
    mol: Chem.Mol | None,
    *,
    strip_salts: bool = False,
    neutralize: bool = False,
) -> Chem.Mol | None:
    """Salt-strip and uncharge an RDKit molecule.

    Returns ``None`` if standardization fails or removes every atom (the
    molecule was a pure counter-ion).
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


def molecule_passes_filter(mol: Chem.Mol | None, config: SmilesFilterConfig) -> bool:
    """Whether ``mol`` passes the size, element, charge, radical, isotope and
    fragment gates of ``config``. Molecules with fewer than three atoms fail.
    """
    try:
        if mol is None:
            return False
        number_of_atoms = mol.GetNumAtoms()
        if number_of_atoms < 3:
            return False
        if config.max_atoms is not None and number_of_atoms > config.max_atoms:
            return False
        if not config.allow_multifragment and (
            Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
            and len(Chem.GetMolFrags(mol, asMols=True)) > 1
        ):
            return False
        if not config.allow_charged and Chem.GetFormalCharge(mol) != 0:
            return False
        return _atoms_pass_filter(mol, config)
    except Exception:
        return False


def _atoms_pass_filter(mol: Chem.Mol, config: SmilesFilterConfig) -> bool:
    """The per-atom gates: element set, radicals, isotopes."""
    allowed_elements = allowed_element_symbols(config.element_set)
    for atom in mol.GetAtoms():
        if allowed_elements is not None and atom.GetSymbol() not in allowed_elements:
            return False
        if not config.allow_radicals and atom.GetNumRadicalElectrons() != 0:
            return False
        if not config.allow_isotopes and atom.GetIsotope() != 0:
            return False
    return True


def _canonical_isomeric_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True)


@dataclass(frozen=True)
class FilteredSmiles:
    """What :func:`filter_smiles` kept, and how many rows it dropped for which reason."""

    #: Positions into the input of the kept rows, in input order.
    kept_row_indices: list[int]
    #: Canonical isomeric SMILES of each kept row.
    isomeric_smiles: list[str]
    #: Canonical SMILES without stereo of each kept row.
    nonisomeric_smiles: list[str]
    #: Unparseable rows, or rows standardization emptied.
    invalid: int
    #: Rows that parsed but failed one of the gates.
    filtered: int
    #: Rows whose canonical isomeric SMILES was already kept (``dedupe`` only).
    duplicates: int


def filter_smiles(
    raw_smiles: Iterable[str | None],
    config: SmilesFilterConfig,
    *,
    seen: set[str] | None = None,
) -> FilteredSmiles:
    """Clean a sequence of raw SMILES; ``None`` entries count as invalid.

    With ``config.dedupe`` the first occurrence of a canonical isomeric SMILES
    wins. ``seen`` is a caller-owned set that carries that memory across calls,
    so a source streamed in priority-ordered batches (train before test) keeps
    first-occurrence-wins semantics across batch boundaries.
    """
    if seen is None:
        seen = set()
    kept_row_indices: list[int] = []
    isomeric: list[str] = []
    nonisomeric: list[str] = []
    invalid = filtered = duplicates = 0
    for row_index, smiles in enumerate(raw_smiles):
        mol = standardize_mol(
            Chem.MolFromSmiles(smiles) if smiles is not None else None,
            strip_salts=config.strip_salts,
            neutralize=config.neutralize,
        )
        if mol is None:
            invalid += 1
            continue
        if not molecule_passes_filter(mol, config):
            filtered += 1
            continue
        canonical = _canonical_isomeric_smiles(mol)
        if config.dedupe:
            if canonical in seen:
                duplicates += 1
                continue
            seen.add(canonical)
        kept_row_indices.append(row_index)
        isomeric.append(canonical)
        nonisomeric.append(Chem.CanonSmiles(canonical, useChiral=False))
    return FilteredSmiles(
        kept_row_indices=kept_row_indices,
        isomeric_smiles=isomeric,
        nonisomeric_smiles=nonisomeric,
        invalid=invalid,
        filtered=filtered,
        duplicates=duplicates,
    )


def standardized_smiles_for_structure(
    mol_implicit_hydrogens: Chem.Mol | None,
    config: SmilesFilterConfig,
    *,
    seen: set[str] | None = None,
) -> str | None:
    """The filter for a source whose molecule already has coordinates.

    Same standardize -> filter -> canonicalize path as :func:`filter_smiles`,
    but a molecule whose standardization would remove heavy atoms (a salt) is
    dropped instead, because its coordinates still describe the unstripped
    molecule. Pass ``seen`` to deduplicate across calls. The input must be in
    implicit-hydrogen form (``Chem.RemoveAllHs`` first if the source supplied
    explicit hydrogens).

    Returns the canonical isomeric SMILES, or ``None`` to drop the molecule.
    """
    if mol_implicit_hydrogens is None:
        return None
    heavy_atoms = mol_implicit_hydrogens.GetNumHeavyAtoms()
    standardized = standardize_mol(
        mol_implicit_hydrogens,
        strip_salts=config.strip_salts,
        neutralize=config.neutralize,
    )
    if standardized is None or standardized.GetNumHeavyAtoms() != heavy_atoms:
        return None
    if not molecule_passes_filter(standardized, config):
        return None
    canonical = _canonical_isomeric_smiles(standardized)
    if config.dedupe and seen is not None:
        if canonical in seen:
            return None
        seen.add(canonical)
    return canonical


__all__ = [
    "FilteredSmiles",
    "SmilesFilterConfig",
    "filter_smiles",
    "molecule_passes_filter",
    "standardize_mol",
    "standardized_smiles_for_structure",
]
