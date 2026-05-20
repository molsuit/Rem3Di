"""MoleculeNet generator — one conformer per SMILES, split materialized.

Driven by a :class:`MoleculeNetBenchmark` from the pydantic registry: reads
the raw release CSV, standardizes/canonicalizes/filters/de-duplicates SMILES,
computes the deterministic DeepChem scaffold split over exactly the kept
SMILES, and emits ``InputBatch`` items carrying targets + per-structure split
codes. The conformer pipeline then generates a single conformer per SMILES.
No cross-zarr SMILES lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from threedscriptors.data_handling.benchmarks import MoleculeNetBenchmark
from threedscriptors.data_handling.dataset.tasks import ElementSet
from threedscriptors.data_handling.dataset_creation.build_stats import LoadStats
from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    filter_mol,
    resolve_element_set,
    standardize_mol,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.splits import (
    deepchem_scaffold_split,
    split_codes,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class MoleculeNetGenerator(MoleculeGenerator):
    def __init__(
        self,
        benchmark: MoleculeNetBenchmark,
        raw_root: Path,
        *,
        batch_size: int = 256,
        max_atoms: int = 100,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        strip_salts: bool = True,
        neutralize: bool = True,
        element_set: ElementSet = ElementSet.mace_off,
    ) -> None:
        self.benchmark = benchmark
        self.csv_path = Path(raw_root) / benchmark.csv_name
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.strip_salts = strip_salts
        self.neutralize = neutralize
        self.element_set = element_set
        self.load_stats = LoadStats()

    def _load(
        self,
    ) -> tuple[list[SmilesData], list[str], np.ndarray, np.ndarray]:
        """Canonicalize/filter/dedupe; return kept SMILES + target/mask matrices."""
        b = self.benchmark
        task_cols = [t.column for t in b.tasks]
        df = pd.read_csv(self.csv_path, usecols=[b.smiles_column, *task_cols])
        targets_all = (
            df[task_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1, len(task_cols))
        )
        smiles_raw = df[b.smiles_column].tolist()

        smiles_data: list[SmilesData] = []
        kept_iso: list[str] = []
        target_rows: list[np.ndarray] = []
        seen: set[str] = set()
        allowed_elements = resolve_element_set(self.element_set)
        stats = self.load_stats
        stats.n_raw_rows = len(smiles_raw)
        for i, smi in enumerate(smiles_raw):
            if smi is None:
                stats.n_invalid_smiles += 1
                continue
            mol = Chem.MolFromSmiles(smi)
            mol = standardize_mol(
                mol, strip_salts=self.strip_salts, neutralize=self.neutralize
            )
            if mol is None:
                stats.n_invalid_smiles += 1
                continue
            if not filter_mol(
                mol, max_atoms=self.max_atoms, allowed_elements=allowed_elements
            ):
                stats.n_filtered_out += 1
                continue
            iso = Chem.MolToSmiles(
                Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True
            )
            if iso in seen:
                stats.n_duplicates += 1
                continue
            seen.add(iso)
            smiles_data.append(
                SmilesData(
                    nonisomeric_smiles=Chem.CanonSmiles(iso, useChiral=False),
                    isomeric_smiles=iso,
                )
            )
            kept_iso.append(iso)
            target_rows.append(targets_all[i])
        stats.n_kept = len(kept_iso)

        targets = np.asarray(target_rows, dtype=np.float64).reshape(
            -1, len(task_cols)
        )
        masks = (~np.isnan(targets)).astype(np.uint8)
        return smiles_data, kept_iso, targets, masks

    def __iter__(self):
        smiles_data, kept_iso, targets, masks = self._load()
        n = len(kept_iso)
        if n == 0:
            return

        train_idx, valid_idx, test_idx = deepchem_scaffold_split(
            kept_iso, self.train_frac, self.val_frac
        )
        codes = split_codes(n, train_idx, valid_idx, test_idx)

        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            structure_ids = [
                StructureID(structure_id=j, molecule_id=j, stereoisomer_id=j)
                for j in range(start, end)
            ]
            yield InputBatch(
                molecules=None,
                smiles=smiles_data[start:end],
                structure_ids=structure_ids,
                regression_data=RegressionData(
                    targets_system=targets[start:end, :],
                    mask_system=masks[start:end, :],
                    split=codes[start:end],
                ),
            )
