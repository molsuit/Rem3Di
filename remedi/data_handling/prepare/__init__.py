"""The prepare manifest — Rem3Di's side of the benchmark build (§3).

``remedi-data`` preparers produce ``smiles``-stage bundles; everything that
happens to a bundle afterwards happens here, as a declarative manifest of
tasks run fault-tolerantly by :func:`prepare`:

* ``generate_conformers`` — bundle(``smiles``) -> bundle(``conformers``), the
  expensive one, and the one whose output gets published (§1.2).
* ``ingest_benchmark`` — bundle(``conformers``) -> zarr, one task per dataset.
* ``verify_benchmark`` — re-open both and re-assert the format (§1.1).

``PrepareContext`` and ``EvalContext`` share the runner
(:mod:`remedi.evaluation.framework.task_runner`), not a base class.
"""

from remedi.data_handling.prepare.config import PrepareManifest, PrepareTask
from remedi.data_handling.prepare.context import BundleRoots, PrepareContext
from remedi.data_handling.prepare.runner import prepare
from remedi.data_handling.prepare.tasks import (
    COPIED_BUNDLE_FILES,
    DEFAULT_GEOMETRY_LIMITS,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    GenerateConformersConfig,
    GenerateConformersSummary,
    IngestBenchmarkConfig,
    IngestBenchmarkSummary,
    PerDatasetPrepareTask,
    VerifyBenchmarkConfig,
    VerifyBenchmarkReport,
)

__all__ = [
    "COPIED_BUNDLE_FILES",
    "DEFAULT_GEOMETRY_LIMITS",
    "STEREO_MISMATCH_AFTER_EMBEDDING",
    "TIMINGS_FILENAME",
    "BundleRoots",
    "GenerateConformersConfig",
    "GenerateConformersSummary",
    "IngestBenchmarkConfig",
    "IngestBenchmarkSummary",
    "PerDatasetPrepareTask",
    "PrepareContext",
    "PrepareManifest",
    "PrepareTask",
    "VerifyBenchmarkConfig",
    "VerifyBenchmarkReport",
    "prepare",
]
