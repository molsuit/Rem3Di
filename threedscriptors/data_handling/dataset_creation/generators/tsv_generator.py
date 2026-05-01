import polars as pl
from rdkit import Chem

from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import filter_mol
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class TSVMoleculeGenerator(MoleculeGenerator):
    def __init__(self, tsv_file: str, batch_size: int):
        self.tsv_file = tsv_file
        self.loading_batch_size = batch_size

    def __iter__(self):
        """
        Generator yielding 'Ligand SMILES' values from a TSV file in streaming batches.
        """
        idx = 0

        smiles_source = pl.scan_csv(
            self.tsv_file, separator="\t", has_header=True
        ).select("Ligand SMILES")

        # 2) Execute in the streaming engine (memory-bounded)
        df = smiles_source.collect(engine="streaming")

        batch_smiles = []
        batch_structure_ids = []

        # 3) Slice into batches and yield one SMILES at a time
        for batch_df in df.iter_slices(n_rows=self.loading_batch_size):
            for smi in batch_df["Ligand SMILES"]:
                if smi is None:
                    continue

                mol = Chem.MolFromSmiles(smi)
                if filter_mol(mol):
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

                if len(batch_smiles) >= self.loading_batch_size:
                    yield InputBatch(
                        molecules=None,
                        smiles=batch_smiles,
                        structure_ids=batch_structure_ids,
                    )
                    batch_smiles, batch_structure_ids = [], []
