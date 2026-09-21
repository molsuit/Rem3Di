"""The prepare manifest — the sibling of ``EvalManifest`` (§3).

One declarative yaml describes a whole prepare run: the three roots, the seed,
and a list of tasks discriminated on ``kind``. It is executed by
:func:`remedi.data_handling.prepare.runner.prepare`, which shares the
per-task loop with the eval runner, so one dataset failing mid-panel is
recorded in ``status.yaml`` and the others still finish.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.bundle import discover_bundles
from remedi.data_handling.prepare.context import BundleRoots
from remedi.data_handling.prepare.tasks import (
    GenerateConformersConfig,
    IngestBenchmarkConfig,
    VerifyBenchmarkConfig,
)

# Discriminated union of prepare-task configs. The manifest and the runner are
# agnostic to the membership: a new kind joins here and is runnable from a
# manifest, and ``expand_tasks`` below expands it per dataset like the rest.
PrepareTask = Annotated[
    GenerateConformersConfig | IngestBenchmarkConfig | VerifyBenchmarkConfig,
    Field(discriminator="kind"),
]


class PrepareManifest(BaseModel):
    """Everything one prepare run needs: three roots x tasks."""

    model_config = ConfigDict(extra="forbid")

    #: Where ``smiles``-stage bundles are read (``remedi-data/bundles``).
    smiles_bundle_root: Path
    #: Where ``conformers``-stage bundles are written and later read.
    benchmark_root: Path
    #: Where zarrs, ``manifest.yaml`` and ``status.yaml`` go.
    output_root: Path

    tasks: list[PrepareTask] = Field(min_length=1)
    seed: int = 0
    # When True (default) a task that raises is recorded in status.yaml and the
    # run continues; set False to fail fast.
    keep_going: bool = True

    def bundle_roots(self) -> BundleRoots:
        """The two roots a task resolves its dataset ids against."""
        return BundleRoots(
            smiles_bundle_root=Path(self.smiles_bundle_root),
            benchmark_root=Path(self.benchmark_root),
        )

    def expand_task(self, task: PrepareTask) -> list[PrepareTask]:
        """One copy of ``task`` per dataset id it covers, resolved *now*.

        A task with ``dataset_ids: null`` covers every bundle under the root it
        reads (``smiles_bundle_root`` for ``generate_conformers``,
        ``benchmark_root`` for ``ingest_benchmark`` / ``verify_benchmark``);
        expanding it into one copy per id is what gives ``status.yaml`` a row
        per dataset instead of one row for the panel.

        **When** this runs matters. ``generate_conformers`` *creates* the
        bundles that the later two kinds discover, so resolving every task's
        ids up front sees only the conformers bundles that happened to exist
        before the run started. :func:`remedi.data_handling.prepare.runner.prepare`
        therefore calls this once per manifest task, immediately before that
        group runs.
        """
        return [
            task.model_copy(update={"dataset_ids": [dataset_id]})
            for dataset_id in task.resolve_dataset_ids(self.bundle_roots())
        ]

    def expand_tasks(self) -> list[PrepareTask]:
        """Every manifest task expanded per dataset, resolved against *today's* disk.

        A preview of the run's shape. The runner does **not** use this — see
        :meth:`expand_task` for why the resolution has to be deferred.
        """
        return [expanded for task in self.tasks for expanded in self.expand_task(task)]

    def resolved_dataset_ids(self) -> list[str]:
        """Every dataset id under ``smiles_bundle_root``, in discovery order."""
        root = Path(self.smiles_bundle_root)
        return [
            directory.relative_to(root).as_posix()
            for directory in discover_bundles(root)
        ]
