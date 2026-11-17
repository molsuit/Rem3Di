"""The per-run context handed to every eval task.

It holds the run's output root and seed, the descriptor model the run
evaluates, and the one descriptor-matrix lookup every task goes through:
:meth:`EvalContext.descriptor_matrix` memoises each matrix for the run, keyed on
the descriptor's identity and the data's (``structures_sha256``), and the
descriptor itself persists it across runs (the ECFP cache, precomputed
embeddings), so a baseline fingerprint or the model's embedding is read or
computed once however many tasks and learners use it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.format import structures_identity
from remedi_eval.descriptors import DescriptorConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    output_root: Path
    #: Where descriptors that compute their own matrices (ECFP) keep them.
    descriptor_cache_dir: Path
    model: DescriptorConfig
    seed: int = 0
    # The manifest's ``keep_going``: False makes a task re-raise the first
    # failure it would otherwise record and step over (fail fast).
    keep_going: bool = True
    _matrices: dict[tuple[str, str], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    def task_dir(self, name: str) -> Path:
        """Create + return ``output_root/<name>`` for a task's artifacts."""
        d = self.output_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def descriptor_matrix(
        self, descriptor: DescriptorConfig, dataset: MoleculeDataset, dataset_id: str
    ) -> np.ndarray:
        """``descriptor``'s ``(N, D)`` matrix over ``dataset`` in zarr order,
        loaded at most once per run for one descriptor and one set of structures.

        Raises:
            FileNotFoundError: if the dataset carries no recorded
                ``structures_sha256`` to key the matrix on.
        """
        key = (descriptor.identity(), structures_identity(Path(dataset.path)))
        if key not in self._matrices:
            logger.info("loading %s over %s", descriptor.name, dataset_id)
            self._matrices[key] = descriptor.load_matrix(
                dataset, dataset_id, self.descriptor_cache_dir
            )
        return self._matrices[key]
