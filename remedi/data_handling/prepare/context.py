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


@dataclass(frozen=True)
class BundleRoots:
    """The two bundle roots a per-dataset prepare task may discover ids from.

    ``generate_conformers`` discovers under ``smiles_bundle_root``;
    ``ingest_benchmark`` and ``verify_benchmark`` under ``benchmark_root``.
    Passing both keeps :meth:`PrepareManifest.expand_tasks` from having to know
    which task reads which root.
    """

    smiles_bundle_root: Path
    benchmark_root: Path


@dataclass
class PrepareContext:
    """Roots and seed for one prepare run."""

    smiles_bundle_root: Path
    benchmark_root: Path
    output_root: Path
    seed: int = 0

    def roots(self) -> BundleRoots:
        """The two bundle roots, for a task resolving its dataset ids."""
        return BundleRoots(
            smiles_bundle_root=self.smiles_bundle_root,
            benchmark_root=self.benchmark_root,
        )

    def task_dir(self, name: str) -> Path:
        """Create + return ``output_root/<name>`` for a task's artifacts."""
        directory = self.output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory
