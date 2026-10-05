"""CXSMILES enhanced stereochemistry -> what a plain SMILES can say.

A CXSMILES may group stereocentres into enhanced-stereo groups:

- ``a`` (absolute): the centre is exactly as drawn.
- ``&n`` (AND): the sample holds both configurations at these centres, usually
  a racemate, so the drawn configuration is arbitrary.
- ``on`` (OR): the sample is one configuration, but which one is unknown.

A plain SMILES has no way to say "and" or "or", and ``Chem.MolToSmiles`` drops
the groups silently, keeping the arbitrary drawn configuration as if it were
known. :func:`unspecify_relative_stereo` instead makes every centre in an AND
or OR group unspecified, so the SMILES claims only what the source knows.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class UnspecifiedStereo:
    """A plain SMILES and how many centres it stopped claiming."""

    smiles: str
    #: Atoms and double bonds made unspecified because they sat in an AND or
    #: OR group; 0 when the input had no such group.
    unspecified_centres: int


def unspecify_relative_stereo(smiles: str) -> UnspecifiedStereo | None:
    """Strip the CXSMILES extension, unspecifying AND/OR-group stereo first.

    Returns ``None`` when RDKit cannot parse ``smiles``; callers pass the
    original string on so the SMILES filter counts it as invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    unspecified = 0
    for group in molecule.GetStereoGroups():
        if group.GetGroupType() == Chem.StereoGroupType.STEREO_ABSOLUTE:
            continue
        for atom in group.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            unspecified += 1
        for bond in group.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            unspecified += 1
    return UnspecifiedStereo(
        smiles=Chem.MolToSmiles(molecule), unspecified_centres=unspecified
    )
