from itertools import chain
from pathlib import Path

from ase import Atoms
from rdkit import Chem

from remedi.configuration.dataset_config import FilterMoleculeStageConfig
from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.generators.utils import (
    standardize_for_conformer,
)
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    SmilesData,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID


class SDFMoleculeGenerator(MoleculeGenerator):
    """Yield raw Atoms + canonical SmilesData straight from an SDF source.

    Coordinate-based gates (size, elements, hydrogen ratio) live in
    ``FilterAtomsStage``. Total charge / spin multiplicity are still computed
    here because SDF carries them per-atom on the Mol and there's nothing to
    recover them from once we've collapsed to Atoms downstream.

    When ``filter_config`` is provided the generator also routes every SMILES
    through the same standardize → filter → canonicalize path as the benchmark
    loaders (``apply_smiles_filter``). The atom-count consistency check inside
    ``standardize_for_conformer`` guarantees the published SMILES still
    describes the molecule whose 3D coordinates we emit. Pass ``None`` (the
    legacy default) to preserve historical behaviour.
    """

    def __init__(
        self,
        sdf_file: Path | list[Path],
        loading_batch_size: int = 100,
        filter_config: FilterMoleculeStageConfig | None = None,
    ):
        self.sdf_file = sdf_file
        self.loading_batch_size = int(loading_batch_size)
        self.filter_config = filter_config
        self._seen: set[str] = set()

    def __iter__(self):
        if isinstance(self.sdf_file, list):
            suppl = chain.from_iterable(
                [Chem.SDMolSupplier(str(p), removeHs=False) for p in self.sdf_file]
            )
        else:
            suppl = Chem.SDMolSupplier(str(self.sdf_file), removeHs=False)

        batch_atoms: list[Atoms] = []
        batch_smiles: list[SmilesData] = []
        batch_structure_ids: list[StructureID] = []
        batch_charges: list[float] = []
        batch_multiplicities: list[float] = []

        for idx, mol in enumerate(suppl):
            if mol is None or mol.GetNumConformers() == 0:
                continue

            if self.filter_config is not None:
                # Route through the same standardization the benchmark loaders
                # use (strip salts / neutralize / element-set filter / canonical
                # isomeric SMILES / dedupe across batches). A None return means
                # the molecule failed one of those gates -- drop it.
                implicit_mol = Chem.RemoveAllHs(mol)
                smiles = standardize_for_conformer(
                    implicit_mol, self.filter_config, seen=self._seen
                )
                if smiles is None:
                    continue
            else:
                smiles = Chem.MolToSmiles(
                    Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True
                )
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            atoms = Atoms(symbols=symbols, positions=pos, info={"smiles": smiles})

            # SDF stores formal charges per atom (and optional M  RAD radical
            # entries). Total charge is the molecule-level sum; spin
            # multiplicity is 2S+1 where 2S equals the total number of
            # unpaired electrons across atoms.
            total_charge = float(Chem.GetFormalCharge(mol))
            n_radical_electrons = sum(
                a.GetNumRadicalElectrons() for a in mol.GetAtoms()
            )
            multiplicity = float(n_radical_electrons + 1)

            batch_atoms.append(atoms)
            batch_smiles.append(
                SmilesData(
                    nonisomeric_smiles=Chem.CanonSmiles(smiles, useChiral=False),
                    isomeric_smiles=smiles,
                )
            )
            batch_structure_ids.append(
                StructureID(structure_id=idx, molecule_id=idx, stereoisomer_id=idx)
            )
            batch_charges.append(total_charge)
            batch_multiplicities.append(multiplicity)

            if len(batch_atoms) >= self.loading_batch_size:
                yield InputBatch(
                    molecules=batch_atoms,
                    smiles=batch_smiles,
                    structure_ids=batch_structure_ids,
                    total_charge=batch_charges,
                    multiplicity=batch_multiplicities,
                )
                batch_atoms, batch_smiles, batch_structure_ids = [], [], []
                batch_charges, batch_multiplicities = [], []

        if batch_atoms:
            yield InputBatch(
                molecules=batch_atoms,
                smiles=batch_smiles,
                structure_ids=batch_structure_ids,
                total_charge=batch_charges,
                multiplicity=batch_multiplicities,
            )
