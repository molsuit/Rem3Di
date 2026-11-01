from dataclasses import dataclass


# Your identifier dataclass
@dataclass(frozen=True, slots=True)
class StructureID:
    structure_id: int  # global unique id for a 3D configuration
    molecule_id: int  # global id for each molecule (shared between isomers of a molecule, defines specific connectivity)
    stereoisomer_id: int  # global  unique id for each stereoisomer, shared by all conformers of that stereoisomer.
