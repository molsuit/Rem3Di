
from rdkit import Chem

from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    filter_mol,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from pathlib import Path


def open_smiles_file(path: Path):

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SmilesMoleculeGenerator(MoleculeGenerator):

    def __init__(self, smiles: list[str], batch_size: int, max_atoms: int):

        self.smiles = smiles
        self.batch_size = batch_size
        self.max_atoms = max_atoms


    def __iter__(self):
        idx = 0

        batch_smiles = []
        batch_structure_ids = []

        # 3) Slice into batches and yield one SMILES at a time
        for i, smi in enumerate(self.smiles):
            if smi is None:
                continue

            mol = Chem.MolFromSmiles(smi)
            if filter_mol(mol, max_atoms=self.max_atoms):
                smiles = Chem.MolToSmiles(
                        Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True
                    )

                batch_smiles.append(
                        SmilesData(
                            nonisomeric_smiles=Chem.CanonSmiles(
                                smiles, useChiral=False
                            ),
                            isomeric_smiles=smiles,
                        )
                    )

                batch_structure_ids.append(
                        StructureID(
                            structure_id=idx, molecule_id=idx, stereoisomer_id=idx
                        )
                    )
                idx += 1

            if len(batch_smiles) >= self.batch_size:
                yield InputBatch(
                    molecules=None,
                    smiles=batch_smiles,
                    structure_ids=batch_structure_ids,
                    regression_data=None
                )
                batch_smiles =[]
                batch_structure_ids = []

        yield InputBatch(
                    molecules=None,
                    smiles=batch_smiles,
                    structure_ids=batch_structure_ids,
                    regression_data=None
                )
