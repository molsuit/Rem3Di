"""The orchestrator: source pickles in, two datasets out.

Stages are chained in memory, so nothing intermediate is written to disk:

1. ``extract``       pickles -> labelled, standardized 3D structures
2. ``validate``      3D stereo audit of the central class + label corrections
3. ``repair``        clash detection, re-embedding and counter-ion stripping
4. ``rebuild``       frozen Architector rebuilds of the flagged organometallics

The result is the usable dataset and the rejected set, each with provenance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from .config import PipelineConfig
from .extract import extract
from .organometallic import rebuild
from .records import BuildResult
from .repair import repair
from .validation import validate
from .writers import (
    write_dataset_extxyz,
    write_dataset_index,
    write_rejected_extxyz,
    write_rejected_index,
    write_run_report,
)

ProgressCallback = Callable[[str], None]


def build_dataset(
    config: PipelineConfig, progress: ProgressCallback | None = None
) -> BuildResult:
    """Run every stage and return the finished dataset plus what was rejected.

    ``progress``, if given, is called with a one-line status after each stage.
    Re-embedding in the repair stage dominates the runtime, so a long build is
    otherwise silent.
    """
    report = progress or (lambda _message: None)
    result = BuildResult()

    started = time.monotonic()
    extraction = extract(config)
    result.rejected.extend(extraction.rejected)
    result.corrections = extraction.corrections
    result.stage_counts["extract"] = extraction.counts
    report(
        f"extract    {len(extraction.structures):>6} structures, "
        f"{len(extraction.rejected):>5} rejected  "
        f"[{time.monotonic() - started:.0f}s]"
    )

    started = time.monotonic()
    validation = validate(config, extraction.structures)
    result.stereo_audit = validation.audit
    result.uncovered_mislabels = validation.uncovered
    result.corrections.extend(validation.corrections)
    result.stage_counts["validate"] = validation.counts
    report(
        f"validate   {validation.counts.get('audited', 0):>6} central audited, "
        f"{len(validation.uncovered):>5} uncovered mislabels  "
        f"[{time.monotonic() - started:.0f}s]"
    )

    started = time.monotonic()
    repaired = repair(config, extraction.structures, extraction.source_lookup)
    result.stage_counts["repair"] = repaired.counts
    report(
        f"repair     {repaired.counts.get('repaired', 0):>6} repaired of "
        f"{repaired.counts.get('clashing', 0)} clashing, "
        f"{repaired.counts.get('unrepaired', 0)} left broken  "
        f"[{time.monotonic() - started:.0f}s]"
    )

    started = time.monotonic()
    rebuilt = rebuild(config, repaired.rejected, progress=report)
    result.stage_counts["organometallic"] = rebuilt.counts
    report(
        f"rebuild    {rebuilt.counts.get('rebuilt', 0):>6} rebuilt of "
        f"{rebuilt.counts.get('targets', 0)} organometallic targets  "
        f"[{time.monotonic() - started:.0f}s]"
    )

    result.structures = sorted(
        repaired.structures + rebuilt.structures, key=lambda s: s.index
    )
    result.rejected.extend(rebuilt.rejected)
    result.rejected.sort(key=lambda r: (r.stage, r.reason, r.index or -1))
    return result


def write_outputs(config: PipelineConfig, result: BuildResult) -> dict[str, Path]:
    """Write both datasets and the run report; return the paths written."""
    output_dir = config.output_dir
    paths = {
        "dataset_index": output_dir / config.output.dataset_index,
        "dataset_extxyz": output_dir / config.output.dataset_extxyz,
        "rejected_index": output_dir / config.output.rejected_index,
        "rejected_extxyz": output_dir / config.output.rejected_extxyz,
        "run_report": output_dir / config.output.run_report,
    }
    write_dataset_index(result.structures, paths["dataset_index"])
    write_dataset_extxyz(result.structures, paths["dataset_extxyz"])
    write_rejected_index(result.rejected, paths["rejected_index"])
    write_rejected_extxyz(result.rejected, paths["rejected_extxyz"])
    write_run_report(
        result, paths["run_report"], config.model_dump(mode="json", exclude={"base_dir"})
    )
    return paths
