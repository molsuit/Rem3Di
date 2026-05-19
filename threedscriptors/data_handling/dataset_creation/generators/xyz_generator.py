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
        max_atoms: int | None = None,
        reject_zero_h: bool = False,
        min_h_heavy_ratio: float = 0.0,
    ):
        self.xyz_file = xyz_file
        self.loading_batch_size = int(loading_batch_size)
        self.charge_key = charge_key
        self.spin_key = spin_key
        self.max_atoms = max_atoms
        self.reject_zero_h = reject_zero_h
        self.min_h_heavy_ratio = float(min_h_heavy_ratio)

        self._n_dropped_size = 0
        self._n_dropped_zero_h = 0
        self._n_dropped_low_h = 0
        self._n_kept = 0

    def filter_systems(self, mol: Atoms) -> bool:
        if self.max_atoms is not None and len(mol) > self.max_atoms:
            self._n_dropped_size += 1
            return False
        nums = mol.get_atomic_numbers()
        n_h = int((nums == 1).sum())
        n_heavy = int((nums > 1).sum())
        if n_heavy == 0:
            self._n_dropped_zero_h += 1
            return False
        if self.reject_zero_h and n_h == 0:
            self._n_dropped_zero_h += 1
            return False
        if (n_h / n_heavy) < self.min_h_heavy_ratio:
            self._n_dropped_low_h += 1
            return False
        return True

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
            suppl = chain.from_iterable([iread(p, index= ":") for p in self.xyz_file])
        else:
            suppl = iread(self.xyz_file, index= ":")

        batch_atoms: list[Atoms] = []
        batch_structure_ids: list[StructureID] = []
        batch_charges: list[float] = []
        batch_multiplicities: list[float] = []

        for idx, atoms in enumerate(suppl):
            if not self.filter_systems(atoms):
                continue

            self._n_kept += 1
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

        # flush tail
        if batch_atoms:
            yield InputBatch(
                molecules=batch_atoms,
                smiles=None,
                structure_ids=batch_structure_ids,
                total_charge=batch_charges,
                multiplicity=batch_multiplicities,
            )

        total_seen = (
            self._n_kept
            + self._n_dropped_size
            + self._n_dropped_zero_h
            + self._n_dropped_low_h
        )
        print(
            f"XYZMoleculeGenerator: kept {self._n_kept}/{total_seen}  "
            f"dropped: size={self._n_dropped_size} "
            f"zero_h={self._n_dropped_zero_h} "
            f"low_h={self._n_dropped_low_h} "
            f"(reject_zero_h={self.reject_zero_h}, min_h_heavy_ratio={self.min_h_heavy_ratio})"
        )
