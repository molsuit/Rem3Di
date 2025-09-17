from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path

import polars as pl
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import Mol

from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID

# Elements supported by your downstream MACE-OFF stack
MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


# ---------- worker-side helpers (must be top-level for pickling) ----------


def filter_mol(mol: Mol, require_3D=False, max_atoms: int | None = None) -> bool:
    """Return True if mol passes all filters, otherwise False."""
    try:
        if mol is None:
            return False
        # must have a 3D conformer
        if require_3D and (mol.GetNumConformers() == 0):
            return False
        # single fragment only
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 3:
            return False
        if max_atoms is not None and num_atoms > max_atoms:
            return False
        if (
            Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
            and len(Chem.GetMolFrags(mol, asMols=True)) > 1
        ):
            return False

        for a in mol.GetAtoms():
            if a.GetSymbol() not in MACE_OFF_ELEMENTS:
                return False
            if a.GetNumRadicalElectrons() != 0:
                return False
            if a.GetIsotope() != 0:
                return False
            if a.GetFormalCharge() != 0:
                return False
        return True
    except Exception:
        return False


class MoleculeGenerator(Iterable[InputBatch], ABC):
    """Base class for molecule generators yielding InputBatch instances.
    Subclasses must implement an efficient ``__iter__`` that yields
    ``InputBatch`` objects, ideally streaming to minimize memory usage.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[InputBatch]:  # pragma: no cover - interface only
        """Return an iterator over ``InputBatch`` items."""
        raise NotImplementedError


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

        # 2) Execute in the streaming engine (memory‐bounded)
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
