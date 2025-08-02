from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructureID:
    structure_id: int  # This is a global and uniqe id of a 3D configuration
    canonical_smiles: str  # The canonical smiles associated to the structure
    molecule_id: int  # The unique id for a molecule, which is identical for enantiomers
    conformer_id: int  # The id that runs from 0 to N_conformer
    enantiomer_id: int | None  = None# None or 1,2 that identifies enantiomers
    smiles_id : int | None = None # Essentially that dataset index (is added separatley because enantiomers have to get the same mol id but have different dataset ids )



    def to_id_string(self) -> str:

        """Human-readable, self-contained ID string *without* smiles_id."""
        string =  (
            f"{self.structure_id}"
            f"-{self.molecule_id}"
            f"-{self.conformer_id}"
            )
        if self.enantiomer_id is not None:
            string = string + f"-{self.enantiomer_id}"

        return string

    @classmethod
    def from_id_string(cls, id_string : str, smiles : str):
        parts =  id_string.split("-", 4)

        if len(parts) == 3:
            structure_id, molecule_id, conformer_id = parts
            enant = None

        elif len(parts) == 4:
            structure_id, molecule_id, conformer_id, enant = parts
            enant = int(enant)

        else:
            raise ValueError("Malformed StructureID string")


        return cls(
            structure_id=int(structure_id),
            canonical_smiles=smiles,
            molecule_id=int(molecule_id),
            conformer_id=int(conformer_id),
            enantiomer_id=enant,
        )
