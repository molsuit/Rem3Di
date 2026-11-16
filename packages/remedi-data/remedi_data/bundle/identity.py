"""The three id levels and the mirror-image relation of §4.2.

``stereoisomer_id`` / ``molecule_id`` / ``enantiomer_of`` are derived from the
canonical isomeric SMILES, which encodes tetrahedral centres and double-bond
geometry only. Axial, helical and planar enantiomers canonicalise to the same
string, share one ``stereoisomer_id`` and keep a null ``enantiomer_of``; that is
accepted (§4.2) because nothing splits or pairs on that identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from rdkit import Chem


class SmilesParseError(ValueError):
    """Raised when RDKit cannot parse a SMILES string."""


class CanonicalSmilesPair(NamedTuple):
    """The two canonical forms one input SMILES resolves to."""

    isomeric: str
    nonisomeric: str


def canonical_smiles_pair(smiles: str) -> CanonicalSmilesPair:
    """Canonicalise ``smiles`` into its isomeric and non-isomeric forms.

    Raises:
        SmilesParseError: if RDKit cannot parse the string.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SmilesParseError(f"RDKit could not parse SMILES {smiles!r}")
    return CanonicalSmilesPair(
        isomeric=Chem.MolToSmiles(molecule),
        nonisomeric=Chem.MolToSmiles(molecule, isomericSmiles=False),
    )


def mirror_isomeric_smiles(isomeric_smiles: str) -> str:
    """Canonical SMILES of the exact mirror image (all tetrahedral centres inverted).

    A molecule that is its own mirror image (achiral, or meso: chiral centres but
    an internal mirror plane) maps to itself, which is how meso compounds end up
    with a null ``enantiomer_of``.

    Raises:
        SmilesParseError: if RDKit cannot parse the string.
    """
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        raise SmilesParseError(f"RDKit could not parse SMILES {isomeric_smiles!r}")
    for atom in molecule.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(molecule)


@dataclass(frozen=True)
class IdentityTable:
    """The identity half of ``table.parquet``: one entry per input SMILES.

    ``stereoisomer_id`` and ``molecule_id`` are dense and numbered in first-seen
    order; ``enantiomer_of`` holds the ``stereoisomer_id`` of the exact mirror
    image when that mirror image is present in the same input, and ``<NA>``
    otherwise.
    """

    stereoisomer_id: np.ndarray
    molecule_id: np.ndarray
    isomeric_smiles: list[str]
    nonisomeric_smiles: list[str]
    enantiomer_of: pd.api.extensions.ExtensionArray

    def __len__(self) -> int:
        return len(self.isomeric_smiles)

    def to_frame(self) -> pd.DataFrame:
        """The fixed leading columns of a bundle with ``smiles: true``, in order."""
        return pd.DataFrame(
            {
                "stereoisomer_id": self.stereoisomer_id,
                "molecule_id": self.molecule_id,
                "enantiomer_of": self.enantiomer_of,
                "isomeric_smiles": pd.array(self.isomeric_smiles, dtype="str"),
                "nonisomeric_smiles": pd.array(self.nonisomeric_smiles, dtype="str"),
            }
        )


def assign_identity(isomeric_smiles: Sequence[str]) -> IdentityTable:
    """Assign dense ids and the mirror-image relation to a sequence of SMILES.

    The input need not be canonical and need not be unique: it is canonicalised
    first, so two spellings of the same molecule collapse onto one
    ``stereoisomer_id``.

    Raises:
        SmilesParseError: if RDKit cannot parse one of the strings.
    """
    canonical = [canonical_smiles_pair(smiles) for smiles in isomeric_smiles]
    canonical_isomeric = [pair.isomeric for pair in canonical]
    mirror_cache: dict[str, str] = {}
    for smiles in canonical_isomeric:
        if smiles not in mirror_cache:
            mirror_cache[smiles] = mirror_isomeric_smiles(smiles)
    return identity_from_canonical(
        canonical_isomeric,
        [pair.nonisomeric for pair in canonical],
        [mirror_cache[smiles] for smiles in canonical_isomeric],
    )


def identity_from_canonical(
    isomeric_smiles: Sequence[str],
    nonisomeric_smiles: Sequence[str],
    mirror_smiles: Sequence[str],
) -> IdentityTable:
    """The ids of :func:`assign_identity` from SMILES that are already canonical.

    No RDKit call: the structure preparers compute the canonical forms and the
    mirror image per shard, in parallel, so the global step is factorising
    only. ``mirror_smiles[i]`` is ``mirror_isomeric_smiles(isomeric_smiles[i])``
    (equal to ``isomeric_smiles[i]`` for a molecule that is its own mirror image).
    """
    isomeric = pd.Series(list(isomeric_smiles), dtype="str")
    stereoisomer_id = pd.factorize(isomeric)[0].astype("int64")
    molecule_id = pd.factorize(pd.Series(list(nonisomeric_smiles), dtype="str"))[
        0
    ].astype("int64")
    first_id_by_smiles = pd.Series(stereoisomer_id, index=isomeric)
    first_id_by_smiles = first_id_by_smiles[~first_id_by_smiles.index.duplicated()]
    mirror = pd.Series(list(mirror_smiles), dtype="str")
    partner = mirror.map(first_id_by_smiles).astype("Int64")
    partner[(mirror == isomeric).to_numpy()] = pd.NA
    # Only mutual pairs count: RDKit keeps meaningless tetrahedral tags on the
    # symmetric atoms of some achiral cages (2-substituted adamantanes), so
    # mirroring A can give B while B is its own mirror image (issues file).
    mirror_by_id = pd.Series(mirror.to_numpy(), index=stereoisomer_id)
    mirror_by_id = mirror_by_id[~mirror_by_id.index.duplicated()]
    has_partner = partner.notna().to_numpy()
    partner_ids = partner[has_partner].astype("int64").to_numpy()
    mutual = (
        mirror_by_id.loc[partner_ids].to_numpy() == isomeric.to_numpy()[has_partner]
    )
    rows_with_partner = np.flatnonzero(has_partner)
    partner[rows_with_partner[~mutual]] = pd.NA
    return IdentityTable(
        stereoisomer_id=stereoisomer_id,
        molecule_id=molecule_id,
        isomeric_smiles=list(isomeric_smiles),
        nonisomeric_smiles=list(nonisomeric_smiles),
        enantiomer_of=partner.array,
    )
