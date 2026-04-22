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
        charge_key: str | None = None,
        spin_key: str | None = None,
    ):
        self.xyz_file = xyz_file
        self.loading_batch_size = int(loading_batch_size)
        self.charge_key = charge_key
        self.spin_key = spin_key

    def _read_charge(self, atoms: Atoms) -> float:
        if self.charge_key is None:
            return 0.0
        return float(atoms.info[self.charge_key])

    def _read_spin(self, atoms: Atoms) -> float:
        if self.spin_key is None:
            return 0.0
        return float(atoms.info[self.spin_key])

    def __iter__(self):
        if isinstance(self.xyz_file, list):
            suppl = chain.from_iterable([iread(p) for p in self.xyz_file])
        else:
            suppl = iread(self.xyz_file)

        batch_atoms: list[Atoms] = []
        batch_structure_ids: list[StructureID] = []
        batch_charges: list[float] = []
        batch_spins: list[float] = []

        for idx, atoms in enumerate(suppl):
            batch_atoms.append(atoms)
            batch_structure_ids.append(
                StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
            )
            batch_charges.append(self._read_charge(atoms))
            batch_spins.append(self._read_spin(atoms))

            if len(batch_atoms) >= self.loading_batch_size:
                yield InputBatch(
                    molecules=batch_atoms,
                    smiles=None,
                    structure_ids=batch_structure_ids,
                    total_charge=batch_charges,
                    total_spin=batch_spins,
                )
                batch_atoms, batch_structure_ids = [], []
                batch_charges, batch_spins = [], []

        # flush tail
        if batch_atoms:
            yield InputBatch(
                molecules=batch_atoms,
                smiles=None,
                structure_ids=batch_structure_ids,
                total_charge=batch_charges,
                total_spin=batch_spins,
            )
