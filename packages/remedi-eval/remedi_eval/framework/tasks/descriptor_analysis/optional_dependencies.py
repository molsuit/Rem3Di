"""The ``descriptor-analysis`` extra: pymatgen (tmQM chemistry) and chemiscope.

Both are imported where they are used, so the analyses that need neither run on
a base install; an analysis that needs one fails with an error naming the extra.
"""

from __future__ import annotations

DESCRIPTOR_ANALYSIS_EXTRA = "descriptor-analysis"


def missing_extra_error(module_name: str, error: ImportError) -> ImportError:
    """The error to raise (``from error``) when ``module_name`` is not installed."""
    return ImportError(
        f"{module_name} is not installed ({error}); this descriptor analysis needs "
        f"the {DESCRIPTOR_ANALYSIS_EXTRA!r} extra: "
        f"pip install 'remedi-eval[{DESCRIPTOR_ANALYSIS_EXTRA}]'"
    )
