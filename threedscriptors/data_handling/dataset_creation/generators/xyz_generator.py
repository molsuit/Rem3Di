from itertools import chain
from pathlib import Path

from ase import Atoms
from ase.io import iread

from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class XYZMoleculeGenerator(MoleculeGenerator):
    def __init__(
        self,
        xyz_file: Path | list[Path],
        loading_batch_size: int = 100,
    ):
        self.xyz_file = xyz_file
        self.loading_batch_size = int(loading_batch_size)

    def __iter__(self):
        if isinstance(self.xyz_file, list):
            suppl = chain.from_iterable([iread(p) for p in self.xyz_file])
        else:
            suppl = iread(self.xyz_file)

        batch_atoms: list[Atoms] = []
        batch_structure_ids: list[StructureID] = []

        for idx, atoms in enumerate(suppl):
            batch_atoms.append(atoms)
            batch_structure_ids.append(
                StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
            )

            if len(batch_atoms) >= self.loading_batch_size:
                yield InputBatch(
                    molecules=batch_atoms,
                    smiles = None,
                    structure_ids=batch_structure_ids,
                )
                batch_atoms, batch_structure_ids = [], []

        # flush tail
        if batch_atoms:
            yield InputBatch(
                molecules=batch_atoms,
                smiles = None,
                structure_ids=batch_structure_ids,
            )
