"""MoleculeNet generator — one conformer per SMILES, split materialized.

Driven by a :class:`MoleculeNetBenchmark` from the pydantic registry: reads the
raw release CSV, runs the shared ``apply_smiles_filter`` helper, computes the
deterministic DeepChem scaffold split over exactly the kept SMILES, and emits
``InputBatch`` items carrying targets + per-structure split codes.

MoleculeNet does not route through ``FilterMoleculeStage`` because the scaffold
split is a *global* operation over the surviving SMILES set, whereas the stage
runs per-batch. We share the filter *logic* via ``apply_smiles_filter`` so the
parse / standardize / canonicalize / dedupe rules stay identical to the rest
of the SMILES sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from threedscriptors.configuration.dataset_config import FilterMoleculeStageConfig
from threedscriptors.data_handling.benchmarks import MoleculeNetBenchmark
from threedscriptors.data_handling.dataset_creation.build_stats import LoadStats
from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.utils import (
    apply_smiles_filter,
    resolve_element_set,
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
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        filter_config: FilterMoleculeStageConfig | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.csv_path = Path(raw_root) / benchmark.csv_name
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.filter_config = filter_config or FilterMoleculeStageConfig()
        self.load_stats = LoadStats()

    def _load(
        self,
    ) -> tuple[list[SmilesData], list[str], np.ndarray, np.ndarray]:
        """Canonicalize / filter / dedupe; return kept SMILES + target/mask matrices."""
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

        cfg = self.filter_config
        allowed_elements = resolve_element_set(cfg.element_set)
        self.load_stats = LoadStats()
        smiles_data, kept_idx = apply_smiles_filter(
            smiles_raw,
            max_atoms=cfg.max_atoms,
            allowed_elements=allowed_elements,
            allow_charged=cfg.allow_charged,
            allow_radicals=cfg.allow_radicals,
            allow_isotopes=cfg.allow_isotopes,
            allow_multifragment=cfg.allow_multifragment,
            strip_salts=cfg.strip_salts,
            neutralize=cfg.neutralize,
            dedupe=cfg.dedupe,
            stats=self.load_stats,
        )

        kept_iso = [sd.isomeric_smiles for sd in smiles_data]
        idx_arr = np.asarray(kept_idx, dtype=np.int64) if kept_idx else np.empty(0, dtype=np.int64)
        targets = targets_all[idx_arr] if kept_idx else np.empty(
            (0, len(task_cols)), dtype=float
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
