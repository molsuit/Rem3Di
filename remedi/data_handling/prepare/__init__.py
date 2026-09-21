"""The prepare manifest — Rem3Di's side of the benchmark build (§3).

``remedi-data`` preparers produce ``smiles``-stage bundles; everything that
happens to a bundle afterwards happens here, as a declarative manifest of
tasks run fault-tolerantly by :func:`prepare`:

* ``generate_conformers`` — bundle(``smiles``) -> bundle(``conformers``), the
  expensive one, and the one whose output gets published (§1.2).
* ``ingest_benchmark`` / ``verify_benchmark`` — build-order step 4.

``PrepareContext`` and ``EvalContext`` share the runner
(:mod:`remedi.evaluation.framework.task_runner`), not a base class.
"""

from remedi.data_handling.prepare.config import PrepareManifest, PrepareTask
from remedi.data_handling.prepare.context import PrepareContext
from remedi.data_handling.prepare.runner import prepare
from remedi.data_handling.prepare.tasks import (
    DEFAULT_GEOMETRY_LIMITS,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    GenerateConformersConfig,
    GenerateConformersSummary,
)

__all__ = [
    "DEFAULT_GEOMETRY_LIMITS",
    "STEREO_MISMATCH_AFTER_EMBEDDING",
    "TIMINGS_FILENAME",
    "GenerateConformersConfig",
    "GenerateConformersSummary",
    "PrepareContext",
    "PrepareManifest",
    "PrepareTask",
    "prepare",
]
