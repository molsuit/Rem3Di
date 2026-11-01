"""Dataset-build machinery: generators, pipeline stages, writer, orchestrator.

The batch dataclasses and the structure-id record are re-exported eagerly
because they are torch-free. ``Pipeline`` / ``PipelineStage`` live in
:mod:`pipeline_stages`, which imports torch, so they are resolved lazily
through ``__getattr__`` and importing the batch dataclasses does not pull in
torch.
"""

from typing import TYPE_CHECKING, Any

from remedi.data_handling.dataset_creation.loading_batch import DataBatch, InputBatch
from remedi.data_handling.dataset_creation.structure_ids import StructureID

if TYPE_CHECKING:  # pragma: no cover - the lazy names, for type checkers
    from remedi.data_handling.dataset_creation.pipeline_stages import (
        Pipeline,
        PipelineStage,
    )

_LAZY_MODULE_BY_NAME = {
    "Pipeline": "remedi.data_handling.dataset_creation.pipeline_stages",
    "PipelineStage": "remedi.data_handling.dataset_creation.pipeline_stages",
}

__all__ = ["DataBatch", "InputBatch", "Pipeline", "PipelineStage", "StructureID"]


def __getattr__(name: str) -> Any:
    """Import the torch-dependent re-exports on first use."""
    module_name = _LAZY_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
