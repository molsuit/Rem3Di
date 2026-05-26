"""Polaris generator that reads a pre-dumped parquet, never importing polaris.

``polaris-lib`` pins ``zarr<3`` and so cannot share a venv with the rest of
this project (which requires ``zarr>=3.2`` for the MoleculeDataset store).
The workaround is a standalone PEP 723 dump script,
``scripts/dataset_download/dump_polaris.py``, that runs in its own ephemeral
env and writes a standardized parquet -- columns ``smiles``, ``split`` (uint8
Split code), and one column per task. This generator reads that parquet and
emits ``InputBatch`` items the same way :class:`TdcGenerator` does. As a
result the package never has to ``import polaris``.

Some polaris datasets (biogen/adme-fang-v1, polaris/drewry2017-pkis2-subset-v2)
ship without a "Set" column, so the dump leaves them entirely unassigned. When
the loaded split codes are all ``Split.unassigned``, this generator applies a
deterministic DeepChem Bemis-Murcko scaffold split before yielding batches --
identical to ``MoleculeNetGenerator``'s split. Mixed states (some assigned,
some not) are passed through as-is so partial annotations are preserved.

The ASAP antiviral datasets ship a train/test split with no validation fold;
when train and test exist but valid is empty, a fixed-seed slice of train is
re-tagged as valid so downstream learners that need an early-stopping signal
have something to use. The polaris test fold is left untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from threedscriptors.data_handling.benchmarks import PolarisBenchmark
from threedscriptors.data_handling.dataset.tasks import Split
from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
)
from threedscriptors.data_handling.dataset_creation.splits import (
    deepchem_scaffold_split,
    split_codes,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID

logger = logging.getLogger(__name__)


class PolarisOfflineGenerator(MoleculeGenerator):
    """Stream rows from a polaris parquet dump in fixed-size batches.

    Schema expected on disk (produced by ``dump_polaris.py``): ``smiles`` (str),
    ``split`` (uint8 :class:`Split` code), one column per
    :class:`BenchmarkTask` whose name matches ``BenchmarkTask.column``. Missing
    cells become NaN and are masked out per-row. When the loaded ``split``
    column is all-unassigned, a deterministic scaffold split is computed at
    iter time -- see module docstring.
    """

    def __init__(
        self,
        benchmark: PolarisBenchmark,
        polaris_raw_root: Path,
        *,
        batch_size: int = 256,
        scaffold_train_frac: float = 0.8,
        scaffold_val_frac: float = 0.1,
        valid_carve_frac: float = 0.1,
        valid_carve_seed: int = 0,
    ) -> None:
        self.benchmark = benchmark
        self.parquet_path = Path(polaris_raw_root) / benchmark.parquet_filename
        self.batch_size = batch_size
        self.scaffold_train_frac = scaffold_train_frac
        self.scaffold_val_frac = scaffold_val_frac
        self.valid_carve_frac = valid_carve_frac
        self.valid_carve_seed = valid_carve_seed

    def _load(self) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Polaris parquet not found: {self.parquet_path}. "
                "Run scripts/dataset_download/dump_polaris.py for "
                f"{self.benchmark.polaris_slug!r} first."
            )
        df = pd.read_parquet(self.parquet_path)
        task_cols = [t.column for t in self.benchmark.tasks]
        missing = [c for c in task_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Polaris parquet {self.parquet_path} missing task columns "
                f"{missing}. Available: {list(df.columns)}"
            )
        smiles = df["smiles"].astype(str).tolist()
        # pyarrow may return Float64 NaN-bearing arrays; coerce to a plain
        # float64 ndarray so the mask path matches TdcGenerator/MoleculeNet.
        targets = (
            df[task_cols].apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1, len(task_cols))
        )
        masks = (~np.isnan(targets)).astype(np.uint8)
        if "split" in df.columns:
            codes = df["split"].to_numpy(dtype=np.uint8)
        else:
            codes = np.full(len(smiles), Split.unassigned.value, dtype=np.uint8)
        codes = self._fill_unassigned_with_scaffold_split(smiles, codes)
        codes = self._carve_valid_from_train_if_missing(codes)
        return smiles, targets, masks, codes

    def _fill_unassigned_with_scaffold_split(
        self, smiles: list[str], codes: np.ndarray
    ) -> np.ndarray:
        """If every row is ``Split.unassigned``, replace codes with a scaffold split.

        Matches :class:`MoleculeNetGenerator`'s split (deterministic
        Bemis-Murcko, ``include_chirality=True``). Mixed-assignment parquets
        are left untouched so partial annotations are respected.
        """
        if len(codes) == 0:
            return codes
        if not np.all(codes == Split.unassigned.value):
            return codes
        logger.info(
            "%s: all rows unassigned; applying scaffold split "
            "(train=%.2f, val=%.2f).",
            self.benchmark.dataset_id,
            self.scaffold_train_frac,
            self.scaffold_val_frac,
        )
        train_idx, valid_idx, test_idx = deepchem_scaffold_split(
            smiles, self.scaffold_train_frac, self.scaffold_val_frac
        )
        return split_codes(len(smiles), train_idx, valid_idx, test_idx)

    def _carve_valid_from_train_if_missing(self, codes: np.ndarray) -> np.ndarray:
        """When the polaris split has test rows but no valid rows, re-tag a
        seeded slice of train as valid so early-stopping has a signal.

        The test fold (and any other already-valid rows) is left alone --
        only train indices move. No-op when the parquet already has a valid
        fold, when there is no test fold, or when ``valid_carve_frac`` is 0.
        """
        if self.valid_carve_frac <= 0.0:
            return codes
        has_test = bool((codes == Split.test.value).any())
        has_valid = bool((codes == Split.valid.value).any())
        if not has_test or has_valid:
            return codes
        train_mask = codes == Split.train.value
        train_idx = np.flatnonzero(train_mask)
        if train_idx.size == 0:
            return codes
        rng = np.random.default_rng(self.valid_carve_seed)
        n_val = max(1, round(train_idx.size * self.valid_carve_frac))
        n_val = min(n_val, train_idx.size - 1)  # keep at least one train row
        picked = rng.choice(train_idx, size=n_val, replace=False)
        codes = codes.copy()
        codes[picked] = Split.valid.value
        logger.info(
            "%s: no valid fold in source; carved %d/%d train rows -> valid "
            "(seed=%d).",
            self.benchmark.dataset_id, n_val, train_idx.size,
            self.valid_carve_seed,
        )
        return codes

    def __iter__(self):
        smiles, targets, masks, codes = self._load()
        n = len(smiles)
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
                smiles=None,
                molecules=None,
                raw_smiles=smiles[start:end],
                structure_ids=structure_ids,
                regression_data=RegressionData(
                    targets_system=targets[start:end, :],
                    mask_system=masks[start:end, :],
                    split=codes[start:end],
                ),
            )
