"""The eval manifest: one model, one output root, a list of tasks.

A manifest is the single declarative yaml for one training run's evaluation. It
binds the descriptor model, where artifacts go, the descriptor cache, and
the tasks to run (an ``Annotated`` discriminated union on ``kind``). The runner
(:func:`framework.runner.run_manifest`) executes the tasks fault-tolerantly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from remedi_eval.descriptors import DescriptorConfig
from remedi_eval.framework.tasks import BenchmarkPanelConfig, DescriptorAnalysisConfig

# Discriminated union of task configs; the runner and manifest are agnostic to
# membership — add a variant here and it is runnable from a manifest.
TaskConfig = Annotated[
    BenchmarkPanelConfig | DescriptorAnalysisConfig,
    Field(discriminator="kind"),
]


class EvalManifest(BaseModel):
    """Full evaluation of one model: model x output root x tasks."""

    model_config = ConfigDict(extra="forbid")

    model: DescriptorConfig
    output_root: Path
    # Where computed descriptors (ECFP) are cached. Defaults to
    # ``output_root.parent / "descriptor_cache"`` so it is a sibling of the
    # per-model output dirs and survives wiping a single model's results.
    descriptor_cache_dir: Path | None = None
    tasks: list[TaskConfig] = Field(default_factory=list, min_length=1)
    seed: int = 0
    # When True (default) a task that raises is recorded in status.yaml and the
    # run continues; set False to fail fast.
    keep_going: bool = True

    def descriptor_cache(self) -> Path:
        return self.descriptor_cache_dir or (
            self.output_root.parent / "descriptor_cache"
        )
