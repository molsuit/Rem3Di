"""Coverage accounting for descriptor benchmarks.

A learned descriptor cannot featurise every molecule in a benchmark. Elements
outside the backbone's training set are rejected, conformer generation fails on
some structures, and the survivors are what actually get scored.

The failure mode this guards against is quiet. If a benchmark reports a metric
over the survivors and nothing else, then a descriptor that reached 43% of a task
produces a number sitting in the same column as one that reached 99%, formatted
identically, with nothing to say they are not comparable. The narrow descriptor
usually looks *better*, because the molecules it dropped are the unusual ones.

Measured example: MACE-OFF24 covers ten elements, so on TDC VDss it reaches
**43.4%** of the test set, and on hERG **61.4%**. A MACE-POLAR descriptor covers
Z = 1–83 and reaches essentially all of both. Those AUROCs are not comparable and
must not be averaged into the same panel mean without saying so.

So: report coverage on every row, and mark rows below a threshold.

.. code-block:: python

    cov = coverage_for(n_evaluated=len(y_pred), n_expected=manifest.n_test)
    row.coverage = cov.fraction
    row.coverage_limited = cov.is_limited
"""

from __future__ import annotations

from dataclasses import dataclass

#: Below this fraction a result is not comparable with a full-coverage one.
#: Chosen to be permissive: it flags the clear cases (VDss at 0.43) rather than
#: policing small conformer-generation losses.
DEFAULT_COVERAGE_THRESHOLD: float = 0.90


@dataclass(frozen=True)
class Coverage:
    """How much of an intended evaluation set was actually scored."""

    n_evaluated: int
    n_expected: int
    threshold: float = DEFAULT_COVERAGE_THRESHOLD

    def __post_init__(self) -> None:
        if self.n_expected < 0 or self.n_evaluated < 0:
            raise ValueError(
                f"counts must be non-negative, got n_evaluated={self.n_evaluated}, "
                f"n_expected={self.n_expected}"
            )
        if self.n_evaluated > self.n_expected:
            raise ValueError(
                f"evaluated {self.n_evaluated} molecules but only {self.n_expected} "
                "were expected — the expected count is probably taken after "
                "filtering rather than before, which would make coverage "
                "unconditionally 1.0 and defeat the point"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    @property
    def fraction(self) -> float:
        """Fraction scored, in ``[0, 1]``. An empty expected set counts as 0."""
        if self.n_expected == 0:
            return 0.0
        return self.n_evaluated / self.n_expected

    @property
    def n_dropped(self) -> int:
        return self.n_expected - self.n_evaluated

    @property
    def is_limited(self) -> bool:
        """True when this result should not be compared with a full-coverage one."""
        return self.fraction < self.threshold

    def describe(self) -> str:
        pct = 100.0 * self.fraction
        note = "  COVERAGE-LIMITED" if self.is_limited else ""
        return (
            f"{self.n_evaluated}/{self.n_expected} molecules ({pct:.1f}%)"
            f"{note}"
        )


def coverage_for(
    *,
    n_evaluated: int,
    n_expected: int,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> Coverage:
    """Build a :class:`Coverage`.

    ``n_expected`` must be counted **before** any element or conformer filtering.
    Taking it after filtering makes coverage 1.0 by construction.
    """
    return Coverage(
        n_evaluated=n_evaluated, n_expected=n_expected, threshold=threshold
    )


def comparable(a: Coverage, b: Coverage) -> bool:
    """Whether two results may be compared directly.

    Both must clear the threshold. Two equally coverage-limited results are still
    not comparable, because they will generally have dropped different molecules.
    """
    return not a.is_limited and not b.is_limited
