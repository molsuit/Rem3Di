import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import batched
from pathlib import Path

from typing import Generator
import numpy as np
from ase import Atoms
from rdkit import Chem

from threedscriptors.data_handling.dataset_creation import StructureID
from threedscriptors.data_handling.dataset_creation.generators import MoleculeGenerator
from threedscriptors.data_handling.dataset_creation.generators.molecule_generators import (
    filter_mol,
)
from collections import deque

from random import shuffle

from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    SmilesData,
)

MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


class GeomGenerator(MoleculeGenerator):
    def __init__(
        self,
        geom_dir: Path,
        boltzman_weight_threshold: float,
        max_atoms: int | None = None,
        loading_batch_size: int = 100,
        max_workers: int = os.cpu_count(),
        shuffle_mols: bool = True,
    ):
        self.geom_dir = geom_dir

        self.max_atoms = max_atoms
        self.boltzmann_weight_threshold = boltzman_weight_threshold
        self.loading_batch_size = loading_batch_size
        self.max_workers = max_workers
        self.shuffle_mols = shuffle_mols

    def get_all_mol_paths(self):
        drugs_file = self.geom_dir / "summary_drugs.json"
        with open(drugs_file) as f:
            drugs_summ = json.load(f)

        rdkit_dir = Path(self.geom_dir)
        existing = {
            str(p.relative_to(rdkit_dir)): p  # key: "drug123/drug123.pkl"
            for p in rdkit_dir.rglob("*.pickle")  # finds files in sub-directories too
        }

        mol_paths = [
            existing[path]
            for sub in drugs_summ.values()
            if (path := sub.get("pickle_path")) in existing
        ]

        if self.shuffle_mols:
            shuffle(mol_paths)

        return mol_paths

    @staticmethod
    def _process_one_file(args):
        """
        Worker: read one pickle, run all filters, and return a list of tuples:
        (mol_id, conf_id, canonical_smiles, atomic_numbers (list[int]), positions (ndarray))
        """
        mol_id, mol_path, boltzmann_weight_threshold, max_atoms = args

        with open(mol_path, "rb") as f:
            dic = pickle.load(f)

        # quick rejects
        if dic.get("charge", 0) != 0:
            return []

        mol = Chem.AddHs(Chem.MolFromSmiles(dic["smiles"]))

        if not filter_mol(mol, max_atoms=max_atoms):
            return []

        conformers = [
            c
            for c in dic["conformers"]
            if c.get("boltzmannweight", 0.0) > boltzmann_weight_threshold
        ]
        if not conformers:
            return []

        can_smiles = Chem.CanonSmiles(dic["smiles"])

        results = []
        for conf_id, conf in enumerate(conformers):
            rd = conf["rd_mol"]  # RDKit Mol with a conformer
            nums = [a.GetAtomicNum() for a in rd.GetAtoms()]
            pos = np.asarray(rd.GetConformer().GetPositions(), dtype=float)
            results.append((mol_id, conf_id, can_smiles, nums, pos))

        return results

    def __iter__(self) -> Generator[InputBatch, None, None]:
        mol_paths = self.get_all_mol_paths()
        structure_idx = 0

        # Tune these two to control how aggressively you load ahead:
        file_batch_size = max(
            self.loading_batch_size * 4, 64
        )  # how many files to process at once
        # You can also add a target buffer size if you want; the while-loop below already enforces fixed yields.

        # FIFO buffers so popping from the front is O(1)
        buf_smiles: deque[SmilesData] = deque()
        buf_mols: deque[Atoms] = deque()
        buf_ids: deque[StructureID] = deque()

        for mol_path_batched in batched(mol_paths, file_batch_size):
            # Dispatch work across processes
            args = [
                (i, p, self.boltzmann_weight_threshold, self.max_atoms)
                for i, p in enumerate(mol_path_batched)
            ]

            raw_results: list[tuple[int, int, str, list[int], np.ndarray]] = []

            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(self._process_one_file, a) for a in args]
                for fut in as_completed(futures):
                    chunk = fut.result()
                    if chunk:
                        raw_results.extend(chunk)

            # Deterministic order within this load step
            raw_results.sort(key=lambda t: (t[0], t[1]))  # (molecule_id, conformer_id)

            # Push everything we just loaded into the buffer
            for mol_id, conf_id, can_smi, nums, pos in raw_results:
                atoms = Atoms(numbers=nums, positions=pos, info={"smiles": can_smi})
                buf_mols.append(atoms)
                buf_ids.append(
                    StructureID(
                        structure_id=structure_idx,
                        molecule_id=mol_id,
                        stereoisomer_id=mol_id,
                    )
                )
                buf_smiles.append(
                    SmilesData(
                        isomeric_smiles=can_smi,
                        nonisomeric_smiles=Chem.CanonSmiles(can_smi, useChiral=0),
                    )
                )
                structure_idx += 1

            # While we have enough in the buffer, yield fixed-size batches
            while len(buf_mols) >= self.loading_batch_size:
                batch_smiles = [
                    buf_smiles.popleft() for _ in range(self.loading_batch_size)
                ]
                batch_mols = [
                    buf_mols.popleft() for _ in range(self.loading_batch_size)
                ]
                batch_ids = [buf_ids.popleft() for _ in range(self.loading_batch_size)]

                yield InputBatch(
                    smiles=batch_smiles, molecules=batch_mols, structure_ids=batch_ids
                )

        # Flush any remainder (set this to `if False` if you want only fixed-size batches)
        if buf_mols:
            yield InputBatch(
                smiles=list(buf_smiles),
                molecules=list(buf_mols),
                structure_ids=list(buf_ids),
            )
