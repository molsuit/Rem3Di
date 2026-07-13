"""Chiral-type classification evaluation.

The legacy retention-time regression relics (``chiral_task`` / ``chiral_eval``)
were retired; the chiral task is now a 5-class chirality-type *classification*
benchmark. The generic multiclass capability lives in the shared benchmark
framework (learners / metrics / runner); this package holds only the
chirality-specific per-class confusion-matrix report.
"""

from remedi.evaluation.chiral.chiral_report import (
    ChiralReportConfig,
    ChiralReportSummary,
)

__all__ = ["ChiralReportConfig", "ChiralReportSummary"]
