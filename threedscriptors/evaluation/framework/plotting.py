"""Decoupled plotter registry: artifact-type -> plotting function(s).

Tasks emit *pure-data* artifacts (:class:`TableResult`, :class:`ArrayResult`,
:class:`PydanticResult`). Plotting is separate: a plotter is registered against
an artifact "kind" (a free-form tag, e.g. ``"benchmark_results"``) and turns a
loaded artifact into one or more :class:`FigureResult`. Because artifacts
round-trip from disk, figures can be regenerated offline (``replot``) without
re-running the eval.

Usage::

    @register_plotter("benchmark_results")
    def _bar(df, out_dir):
        ...
        return [FigureResult(file_name=Path("bench.png"), figure=fig)]

    figs = render("benchmark_results", df, out_dir)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from threedscriptors.evaluation.results import FigureResult

logger = logging.getLogger(__name__)

# A plotter maps (loaded artifact payload, output dir) -> list of figure results.
Plotter = Callable[[Any, Path], "list[FigureResult]"]

_PLOTTERS: dict[str, list[Plotter]] = {}


def register_plotter(artifact_kind: str) -> Callable[[Plotter], Plotter]:
    """Decorator: register ``fn`` as a plotter for ``artifact_kind``."""

    def deco(fn: Plotter) -> Plotter:
        _PLOTTERS.setdefault(artifact_kind, []).append(fn)
        return fn

    return deco


def registered_kinds() -> list[str]:
    return sorted(_PLOTTERS)


def render(artifact_kind: str, payload: Any, out_dir: Path) -> list[FigureResult]:
    """Run every plotter registered for ``artifact_kind`` over ``payload``.

    A plotter that raises is logged and skipped — a broken plot must not lose
    the others or the underlying data artifact.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[FigureResult] = []
    for fn in _PLOTTERS.get(artifact_kind, []):
        try:
            produced = fn(payload, out_dir)
        except Exception:
            logger.exception("plotter %s for %s failed; skipping", fn, artifact_kind)
            continue
        for fig in produced:
            fig.serialize_to(out_dir)
            figures.append(fig)
    if artifact_kind not in _PLOTTERS:
        logger.debug("no plotters registered for artifact kind %s", artifact_kind)
    return figures
