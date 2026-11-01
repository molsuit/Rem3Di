import polars as pl

from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.loading_batch import InputBatch
from remedi.data_handling.dataset_creation.structure_ids import StructureID


class TSVMoleculeGenerator(MoleculeGenerator):
    """Stream raw 'Ligand SMILES' rows from a TSV; filtering is downstream."""

    def __init__(self, tsv_file: str, batch_size: int):
        self.tsv_file = tsv_file
        self.loading_batch_size = batch_size

    def __iter__(self):
        smiles_source = pl.scan_csv(
            self.tsv_file, separator="\t", has_header=True
        ).select("Ligand SMILES")
        df = smiles_source.collect(engine="streaming")

        idx = 0
        for batch_df in df.iter_slices(n_rows=self.loading_batch_size):
            raw = [smi for smi in batch_df["Ligand SMILES"] if smi is not None]
            if not raw:
                continue
            structure_ids = [
                StructureID(
                    structure_id=idx + j, molecule_id=idx + j, stereoisomer_id=idx + j
                )
                for j in range(len(raw))
            ]
            idx += len(raw)
            yield InputBatch(
                smiles=None,
                molecules=None,
                raw_smiles=raw,
                structure_ids=structure_ids,
            )
