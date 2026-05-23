from itertools import chain
from pathlib import Path

from ase import Atoms
from ase.io import iread

from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import InputBatch
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class XYZMoleculeGenerator(MoleculeGenerator):
    """Stream raw Atoms from one-or-many extxyz files.

    Size / hydrogen-coverage / element-set gates live in ``FilterAtomsStage``;
    this generator's only job is parsing + lifting ``charge_key`` /
    ``spin_key`` out of the comment line. ``spin_key`` already holds the spin
    multiplicity (2S+1) per OMol25 convention and is read verbatim.
    """

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

    def _read_multiplicity(self, atoms: Atoms) -> float:
        """Return the spin multiplicity (2S+1) for one molecule.

        ``spin_key`` names a field on ``atoms.info`` that already holds the
        spin multiplicity (2S+1), not the total spin S. This matches the
        OMol25 convention, where ``spin`` is the multiplicity passed to ORCA
        (verified against ``s_squared``: ``spin=1`` ↔ ⟨S²⟩=0 singlet,
        ``spin=2`` ↔ ⟨S²⟩≈0.75 doublet) and is exactly the quantity MACE /
        PolarMACE consume. The value is read through verbatim. When no
        ``spin_key`` is configured we fall back to multiplicity 1.0 (closed
        shell), matching MACE's documented default.
        """
        if self.spin_key is None:
            return 1.0
        return float(atoms.info[self.spin_key])

    def __iter__(self):
        if isinstance(self.xyz_file, list):
            suppl = chain.from_iterable([iread(p, index=":") for p in self.xyz_file])
        else:
            suppl = iread(self.xyz_file, index=":")

        batch_atoms: list[Atoms] = []
        batch_structure_ids: list[StructureID] = []
        batch_charges: list[float] = []
        batch_multiplicities: list[float] = []

        for idx, atoms in enumerate(suppl):
            batch_atoms.append(atoms)
            batch_structure_ids.append(
                StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
            )
            batch_charges.append(self._read_charge(atoms))
            batch_multiplicities.append(self._read_multiplicity(atoms))

            if len(batch_atoms) >= self.loading_batch_size:
                yield InputBatch(
                    molecules=batch_atoms,
                    smiles=None,
                    structure_ids=batch_structure_ids,
                    total_charge=batch_charges,
                    multiplicity=batch_multiplicities,
                )
                batch_atoms, batch_structure_ids = [], []
                batch_charges, batch_multiplicities = [], []

        if batch_atoms:
            yield InputBatch(
                molecules=batch_atoms,
                smiles=None,
                structure_ids=batch_structure_ids,
                total_charge=batch_charges,
                multiplicity=batch_multiplicities,
            )
