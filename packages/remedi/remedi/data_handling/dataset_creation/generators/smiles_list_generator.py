from pathlib import Path

from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.loading_batch import InputBatch
from remedi.data_handling.dataset_creation.structure_ids import StructureID


def open_smiles_file(path: Path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SmilesMoleculeGenerator(MoleculeGenerator):
    """Yield raw SMILES batches; filtering is owned by ``FilterMoleculeStage``."""

    def __init__(self, smiles: list[str], batch_size: int):
        self.smiles = smiles
        self.batch_size = batch_size

    def __iter__(self):
        bs = self.batch_size
        n = len(self.smiles)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            raw: list[str | None] = list(self.smiles[start:end])
            structure_ids = [
                StructureID(structure_id=j, molecule_id=j, stereoisomer_id=j)
                for j in range(start, end)
            ]
            yield InputBatch(
                smiles=None,
                molecules=None,
                raw_smiles=raw,
                structure_ids=structure_ids,
            )
