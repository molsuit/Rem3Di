"""ChiralCat chirality-classification dataset ingestion pipeline.

Build the dataset from the source pickles with a single call::

    from pathlib import Path
    from chiralcat_dataset import PipelineConfig, build_dataset, write_outputs

    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    result = build_dataset(config)
    write_outputs(config, result)
"""

from .config import PipelineConfig
from .pipeline import build_dataset, write_outputs
from .records import BuildResult, RejectedRecord, Structure
from .taxonomy import CLASS_ORDER, CLASS_TO_LABEL, normalize_class

__all__ = [
    "CLASS_ORDER",
    "CLASS_TO_LABEL",
    "BuildResult",
    "PipelineConfig",
    "RejectedRecord",
    "Structure",
    "build_dataset",
    "normalize_class",
    "write_outputs",
]
