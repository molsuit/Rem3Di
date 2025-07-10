from dataclasses import dataclass


@dataclass(frozen=True, slots = True)
class StructureID:
    structure_id: int
    canonical_smiles : str
    molecule_id: int
    conformer_id: int
    enantiomer_id: int | None
