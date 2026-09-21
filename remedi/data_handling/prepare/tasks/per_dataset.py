"""The shape every prepare task shares: one task, one panel, many datasets.

All three prepare task kinds (``generate_conformers``, ``ingest_benchmark``,
``verify_benchmark``) run over a list of dataset ids, default to *every* bundle
under one of the run's roots, and are expanded by
:meth:`remedi.data_handling.prepare.config.PrepareManifest.expand_tasks` into
one task per dataset so that one failure is one failed entry in ``status.yaml``
and the rest of the panel still finishes (``BENCHMARK_DATA_FORMAT.md`` §3).

That expansion is the only reason this base class exists: it is what lets the
manifest and the runner stay agnostic to which kinds are in the union.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from remedi.data_handling.bundle import discover_bundles
from remedi.data_handling.prepare.context import BundleRoots


class PerDatasetPrepareTask(BaseModel):
    """A prepare task that runs over one or more bundle dataset ids."""

    model_config = ConfigDict(extra="forbid")

    #: ``None`` means every bundle under this task's :meth:`discovery_root`.
    dataset_ids: list[str] | None = None

    @property
    def status_label(self) -> str | None:
        """The dataset id, once the manifest has expanded this task onto one.

        ``run_tasks`` appends it to the ``status.yaml`` entry name, so the file
        answers "which of the 30 built" without reading a traceback.
        """
        if self.dataset_ids is not None and len(self.dataset_ids) == 1:
            return self.dataset_ids[0].replace("/", "__")
        return None

    def discovery_root(self, roots: BundleRoots) -> Path:
        """The root whose bundles this task covers when ``dataset_ids`` is unset.

        Raises:
            NotImplementedError: if a concrete task kind forgets to declare it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must declare which bundle root it reads"
        )

    def resolve_dataset_ids(self, roots: BundleRoots) -> list[str]:
        """The dataset ids this task covers, discovering them when unset."""
        if self.dataset_ids is not None:
            return list(self.dataset_ids)
        root = Path(self.discovery_root(roots))
        return [
            directory.relative_to(root).as_posix()
            for directory in discover_bundles(root)
        ]
