"""The per-run context handed to every prepare task (``BENCHMARK_DATA_FORMAT.md`` §3).

``PrepareContext`` and ``EvalContext`` do **not** share a base class — they
share the runner (:mod:`remedi.evaluation.framework.task_runner`). An eval run
is about one descriptor model; a prepare run is about three directory roots and
carries no model.

The three roots are distinct on purpose, because the prepare stage moves data
between two repositories:

``smiles_bundle_root``
    where ``smiles``-stage bundles are read — ``remedi-data/bundles``, written
    by the preparers.
``benchmark_root``
    where ``conformers``-stage bundles are written by ``generate_conformers``
    and read back by the ingest task. This is the published artifact (§1.2).
``output_root``
    where zarrs and the run files (``manifest.yaml`` / ``status.yaml``) go.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepareContext:
    """Roots and seed for one prepare run."""

    smiles_bundle_root: Path
    benchmark_root: Path
    output_root: Path
    seed: int = 0

    def task_dir(self, name: str) -> Path:
        """Create + return ``output_root/<name>`` for a task's artifacts."""
        directory = self.output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory
