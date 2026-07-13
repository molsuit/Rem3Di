"""The per-run context handed to every eval task.

Generalises ``descriptor_analysis.DescriptorAnalysisContext``: it owns the
:class:`ResourceCache` (so tasks share embeddings / indices), the run's output
root and seed, and the descriptor model the run evaluates. Tasks pull shared
inputs via ``ctx.resources.get(spec)`` and write into ``ctx.task_dir(name)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from remedi.evaluation.benchmark.descriptors import DescriptorConfig
from remedi.evaluation.framework.resources import ResourceCache


@dataclass
class EvalContext:
    output_root: Path
    resource_cache_dir: Path
    resources: ResourceCache
    model: DescriptorConfig
    seed: int = 0

    def task_dir(self, name: str) -> Path:
        """Create + return ``output_root/<name>`` for a task's artifacts."""
        d = self.output_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d
