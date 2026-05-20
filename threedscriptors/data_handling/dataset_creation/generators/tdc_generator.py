"""TDC ADMET generator — one conformer per SMILES, official split materialized.

Driven by a :class:`TdcBenchmark` from the pydantic registry. Uses PyTDC's
``admet_group`` so the train/valid/test partition is the *official* leaderboard
split, not a recomputed one: the scaffold ``test`` fold is fixed, and
``train_val`` is divided into train/valid by ``get_train_valid_split`` at the
default seed (eval may re-derive valid per seed). Each kept SMILES is ingested
once with its split code inline — no cross-zarr SMILES lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from threedscriptors.data_handling.benchmarks import TdcBenchmark
from threedscriptors.data_handling.dataset.tasks import ElementSet, Split
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
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class TdcGenerator(MoleculeGenerator):
    def __init__(
        self,
        benchmark: TdcBenchmark,
        tdc_cache: Path,
        *,
        batch_size: int = 256,
        max_atoms: int = 100,
        seed: int = 1,
        strip_salts: bool = True,
        neutralize: bool = True,
        element_set: ElementSet = ElementSet.mace_off,
    ) -> None:
        self.benchmark = benchmark
        self.tdc_cache = Path(tdc_cache)
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.seed = seed
        self.strip_salts = strip_salts
        self.neutralize = neutralize
        self.element_set = element_set
        self.load_stats = LoadStats()

    def _split_frames(self) -> list[tuple[pd.DataFrame, Split]]:
        """Official PyTDC partition: seeded train/valid + the fixed scaffold test."""
        from tdc.benchmark_group import admet_group

        group = admet_group(path=str(self.tdc_cache))
        name = self.benchmark.tdc_name
        test_df = group.get(name)["test"]
        train_df, valid_df = group.get_train_valid_split(
            seed=self.seed, benchmark=name, split_type="default"
        )
        return [
            (train_df, Split.train),
            (valid_df, Split.valid),
            (test_df, Split.test),
        ]

    def _load(
        self,
    ) -> tuple[list[SmilesData], np.ndarray, np.ndarray, np.ndarray]:
        """Canonicalize/filter/dedupe across train->valid->test (first wins)."""
        smiles_data: list[SmilesData] = []
        targets: list[float] = []
        codes: list[int] = []
        seen: set[str] = set()
        allowed_elements = resolve_element_set(self.element_set)
        stats = self.load_stats

        for frame, split in self._split_frames():
            stats.n_raw_rows += len(frame)
            for smi, y in zip(frame["Drug"], frame["Y"], strict=True):
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
                targets.append(float(y))
                codes.append(int(split.value))

        stats.n_kept = len(smiles_data)
        targets_arr = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        masks = (~np.isnan(targets_arr)).astype(np.uint8)
        codes_arr = np.asarray(codes, dtype=np.uint8)
        return smiles_data, targets_arr, masks, codes_arr

    def __iter__(self):
        smiles_data, targets, masks, codes = self._load()
        n = len(smiles_data)
        if n == 0:
            return

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
