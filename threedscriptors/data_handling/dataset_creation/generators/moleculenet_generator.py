
import numpy as np
import polars as pl
from rdkit import Chem

from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    filter_mol,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class MoleculeNetGenerator(MoleculeGenerator):
    def __init__(self, file: str, batch_size: int, tasks, mol_column: str, max_atoms):
        self.file = file
        self.loading_batch_size = batch_size
        self.target_cols = tasks
        self.mol_column = mol_column
        self.max_atoms = max_atoms

    def __iter__(self):
        """
        Generator yielding 'Ligand SMILES' values from a TSV file in streaming batches.
        """
        idx = 0
        B = self.loading_batch_size

        smiles_source = pl.scan_csv(self.file, separator=",", has_header=True)

        # 2) Execute in the streaming engine (memory‐bounded)
        df = smiles_source.collect(engine="streaming")

        batch_smiles = []
        batch_structure_ids = []
        targets_buffer = []
        masks_buffer = []

        # 3) Slice into batches and yield one SMILES at a time
        for batch_df in df.iter_slices(n_rows=self.loading_batch_size):
            accept_indices = []


            for i, smi in enumerate(batch_df[self.mol_column]):

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

                    accept_indices.append(i)


            if accept_indices:

                accept_indices = np.array(accept_indices)
                sub = batch_df.select(self.target_cols)
                sub = sub.with_columns(pl.all().cast(pl.Float32))
                t = sub.fill_null(np.nan).to_numpy().reshape(-1,len(self.target_cols))  # (k, n_targets), floats with NaN
                m = sub.select(pl.all().is_not_null()).to_numpy().astype(np.int8).reshape(-1,len(self.target_cols))  # (k, n_targets), 0/1

                targets_buffer.extend(t[accept_indices,:])
                masks_buffer.extend(m[accept_indices,:])

            if len(batch_smiles) >= self.loading_batch_size:

                regression_targets = np.vstack(targets_buffer)
                regression_masks = np.vstack(masks_buffer)


                yield InputBatch(
                    molecules=None,
                    smiles=batch_smiles,
                    structure_ids=batch_structure_ids,
                    regression_data=RegressionData(
                        targets_system=regression_targets, mask_system=regression_masks
                    ),
                )

                batch_smiles = batch_smiles[self.loading_batch_size:]
                batch_structure_ids = batch_structure_ids[self.loading_batch_size :]
                targets_buffer = [
                    regression_targets[self.loading_batch_size :, :]
                ]
                masks_buffer = [
                    regression_masks[self.loading_batch_size :, :]
                ]

        if batch_smiles:

            regression_targets = np.concatenate(targets_buffer)
            regression_masks = np.concatenate(masks_buffer)

            yield InputBatch(
                molecules=None,
                smiles=batch_smiles,
                structure_ids=batch_structure_ids,
                regression_data=RegressionData(
                    targets_system=regression_targets,
                    mask_system=regression_masks
                ),
            )
