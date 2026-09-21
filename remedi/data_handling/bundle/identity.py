"""The three id levels and the mirror-image relation of §1.1.

``stereoisomer_id`` / ``molecule_id`` / ``enantiomer_of`` are derived from the
canonical isomeric SMILES, which encodes tetrahedral centres and double-bond
geometry only. Axial, helical and planar enantiomers canonicalise to the same
string, share one ``stereoisomer_id`` and keep a null ``enantiomer_of``; that is
accepted (§1.1) because nothing splits or pairs on that identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from rdkit import Chem

from remedi.data_handling.bundle.spec import FIXED_COLUMNS


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
        """The six fixed leading columns of §1.1, in order, with ``structure_id``."""
        frame = pd.DataFrame(
            {
                "structure_id": np.arange(len(self), dtype="int64"),
                "stereoisomer_id": self.stereoisomer_id,
                "molecule_id": self.molecule_id,
                "isomeric_smiles": pd.array(self.isomeric_smiles, dtype="str"),
                "nonisomeric_smiles": pd.array(self.nonisomeric_smiles, dtype="str"),
                "enantiomer_of": self.enantiomer_of,
            }
        )
        return frame[list(FIXED_COLUMNS)]


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
    canonical_nonisomeric = [pair.nonisomeric for pair in canonical]

    stereoisomer_id = pd.factorize(pd.Series(canonical_isomeric, dtype="str"))[
        0
    ].astype("int64")
    molecule_id = pd.factorize(pd.Series(canonical_nonisomeric, dtype="str"))[0].astype(
        "int64"
    )

    stereoisomer_id_by_smiles = {
        smiles: int(identifier)
        for smiles, identifier in zip(canonical_isomeric, stereoisomer_id, strict=True)
    }
    partner_ids: list[int | None] = []
    mirror_cache: dict[str, str] = {}
    for smiles in canonical_isomeric:
        if smiles not in mirror_cache:
            mirror_cache[smiles] = mirror_isomeric_smiles(smiles)
        mirror = mirror_cache[smiles]
        if mirror == smiles or mirror not in stereoisomer_id_by_smiles:
            partner_ids.append(None)
        else:
            partner_ids.append(stereoisomer_id_by_smiles[mirror])

    return IdentityTable(
        stereoisomer_id=stereoisomer_id,
        molecule_id=molecule_id,
        isomeric_smiles=canonical_isomeric,
        nonisomeric_smiles=canonical_nonisomeric,
        enantiomer_of=pd.array(partner_ids, dtype="Int64"),
    )
