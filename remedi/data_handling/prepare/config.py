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
from remedi.data_handling.prepare.tasks.generate_conformers import (
    GenerateConformersConfig,
)

# Discriminated union of prepare-task configs. A one-member union today;
# ``ingest_benchmark`` and ``verify_benchmark`` (§3) join it at build-order
# step 4, and the manifest and runner stay agnostic to the membership.
PrepareTask = Annotated[
    GenerateConformersConfig,
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

    def expand_tasks(self) -> list[PrepareTask]:
        """One task per resolved dataset id, so failures isolate per dataset.

        A task with ``dataset_ids: null`` covers every bundle under
        ``smiles_bundle_root``; expanding it into one copy per id is what gives
        ``status.yaml`` a row per dataset instead of one row for the panel.
        """
        expanded: list[PrepareTask] = []
        for task in self.tasks:
            for dataset_id in task.resolve_dataset_ids(self.smiles_bundle_root):
                expanded.append(task.model_copy(update={"dataset_ids": [dataset_id]}))
        return expanded

    def resolved_dataset_ids(self) -> list[str]:
        """Every dataset id under ``smiles_bundle_root``, in discovery order."""
        root = Path(self.smiles_bundle_root)
        return [
            directory.relative_to(root).as_posix()
            for directory in discover_bundles(root)
        ]
