"""The prepare task kinds (``BENCHMARK_DATA_FORMAT.md`` §3).

``generate_conformers`` (bundle -> bundle), ``ingest_benchmark``
(bundle -> zarr) and ``verify_benchmark`` (re-assert both). All three share
:class:`PerDatasetPrepareTask`, which is what lets the manifest expand one task
per dataset and isolate a failure to that dataset.
"""

from remedi.data_handling.prepare.tasks.generate_conformers import (
    DEFAULT_GEOMETRY_LIMITS,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    GenerateConformersConfig,
    GenerateConformersSummary,
)
from remedi.data_handling.prepare.tasks.ingest_benchmark import (
    COPIED_BUNDLE_FILES,
    IngestBenchmarkConfig,
    IngestBenchmarkSummary,
)
from remedi.data_handling.prepare.tasks.per_dataset import PerDatasetPrepareTask
from remedi.data_handling.prepare.tasks.verify_benchmark import (
    VerifyBenchmarkConfig,
    VerifyBenchmarkReport,
)

__all__ = [
    "COPIED_BUNDLE_FILES",
    "DEFAULT_GEOMETRY_LIMITS",
    "STEREO_MISMATCH_AFTER_EMBEDDING",
    "TIMINGS_FILENAME",
    "GenerateConformersConfig",
    "GenerateConformersSummary",
    "IngestBenchmarkConfig",
    "IngestBenchmarkSummary",
    "PerDatasetPrepareTask",
    "VerifyBenchmarkConfig",
    "VerifyBenchmarkReport",
]
