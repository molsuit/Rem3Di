"""TDC ADMET generator — one conformer per SMILES, official split materialized.

Driven by a :class:`TdcBenchmark` from the pydantic registry. Uses PyTDC's
``admet_group`` so the train/valid/test partition is the *official* leaderboard
split, not a recomputed one: the scaffold ``test`` fold is fixed, and
``train_val`` is divided into train/valid by ``get_train_valid_split`` at the
default seed (eval may re-derive valid per seed).

The generator yields raw rows in priority order (train → valid → test) with the
split label inline; canonicalize / filter / dedupe is owned by
``FilterMoleculeStage``. Because the stage's dedupe set persists across batches,
the "first split wins" semantics carry over even though the generator no longer
holds the seen-set itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from remedi.data_handling.benchmarks import TdcBenchmark
from remedi.data_handling.dataset.tasks import Split
from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID

logger = logging.getLogger(__name__)


def _patch_tdc_print_sys() -> None:
    """Inject ``print_sys`` into ``tdc.utils.split``.

    PyTDC 0.x (incl. 0.3.6) catches RDKit RuntimeErrors from
    ``MurckoScaffoldSmiles`` inside ``create_scaffold_split`` and tries to log
    a skip message via ``print_sys(...)`` — but ``print_sys`` is never
    imported into ``tdc.utils.split`` so the handler itself raises
    ``NameError``, aborting the whole build (seen on CYP2C9_Veith with
    a 'bad bond stereo' SMILES). Provide the symbol so the intended
    skip-and-continue runs.
    """
    import tdc.utils.split as _tdc_split

    if hasattr(_tdc_split, "print_sys"):
        return
    try:
        from tdc.utils import print_sys as _print_sys
    except ImportError:

        def _print_sys(msg: str, *args: object, **kwargs: object) -> None:
            logger.warning("tdc skipped: %s", msg)

    # setattr keeps ty happy — print_sys is intentionally not declared on
    # tdc.utils.split (the bug we're patching).
    setattr(_tdc_split, "print_sys", _print_sys)  # noqa: B010


class TdcGenerator(MoleculeGenerator):
    def __init__(
        self,
        benchmark: TdcBenchmark,
        tdc_cache: Path,
        *,
        batch_size: int = 256,
        seed: int = 1,
    ) -> None:
        self.benchmark = benchmark
        self.tdc_cache = Path(tdc_cache)
        self.batch_size = batch_size
        self.seed = seed

    def _split_frames(self) -> list[tuple[pd.DataFrame, Split]]:
        """Official PyTDC partition: seeded train/valid + the fixed scaffold test."""
        from tdc.benchmark_group import admet_group

        _patch_tdc_print_sys()
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

    def _iter_rows(self):
        """Flatten train→valid→test into ``(smiles, target, split_code)`` rows.

        Keeping train first means FilterMoleculeStage's cross-batch dedupe set
        records the train occurrence first; a later valid/test duplicate is
        dropped, preserving the historical "first split wins" priority.
        """
        for frame, split in self._split_frames():
            for smi, y in zip(frame["Drug"], frame["Y"], strict=True):
                yield smi, float(y), int(split.value)

    def __iter__(self):
        bs = self.batch_size
        smi_buf: list[str | None] = []
        target_buf: list[float] = []
        code_buf: list[int] = []
        structure_idx = 0

        def _emit():
            nonlocal structure_idx
            n = len(smi_buf)
            targets = np.asarray(target_buf, dtype=np.float64).reshape(-1, 1)
            masks = (~np.isnan(targets)).astype(np.uint8)
            codes = np.asarray(code_buf, dtype=np.uint8)
            structure_ids = [
                StructureID(
                    structure_id=structure_idx + i,
                    molecule_id=structure_idx + i,
                    stereoisomer_id=structure_idx + i,
                )
                for i in range(n)
            ]
            structure_idx += n
            return InputBatch(
                smiles=None,
                molecules=None,
                raw_smiles=list(smi_buf),
                structure_ids=structure_ids,
                regression_data=RegressionData(
                    targets_system=targets,
                    mask_system=masks,
                    split=codes,
                ),
            )

        for smi, y, code in self._iter_rows():
            smi_buf.append(smi)
            target_buf.append(y)
            code_buf.append(code)
            if len(smi_buf) >= bs:
                yield _emit()
                smi_buf, target_buf, code_buf = [], [], []

        if smi_buf:
            yield _emit()
