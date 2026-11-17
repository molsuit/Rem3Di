"""Datasets: the table files with ``structures.atp`` beside them (§4.1, §6).

:func:`write_dataset` is the one writer every structure dataset goes through;
:func:`build_datasets` turns SMILES bundles into datasets by generating
conformers, then verifies every dataset
(:func:`remedi_data.format.verify.verify_dataset`); :func:`expand_to_structures`
turns a bundle's rows into one row per structure.
"""

from remedi_data.build.build import (
    CONFORMER_EMBEDDING_FAILED,
    DEFAULT_GEOMETRY_LIMITS,
    MMFF_NOT_APPLIED,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    BuildDatasetTask,
    BuildSummary,
    DatasetBuildConfig,
    VerifyDatasetTask,
    build_datasets,
    embed_bundle,
)
from remedi_data.build.expansion import (
    ENANTIOMER_PARTNER_FAILED,
    ExpandedDataset,
    expand_to_structures,
)
from remedi_data.build.write import write_dataset

__all__ = [
    "CONFORMER_EMBEDDING_FAILED",
    "DEFAULT_GEOMETRY_LIMITS",
    "ENANTIOMER_PARTNER_FAILED",
    "MMFF_NOT_APPLIED",
    "STEREO_MISMATCH_AFTER_EMBEDDING",
    "TIMINGS_FILENAME",
    "BuildDatasetTask",
    "BuildSummary",
    "DatasetBuildConfig",
    "ExpandedDataset",
    "VerifyDatasetTask",
    "build_datasets",
    "embed_bundle",
    "expand_to_structures",
    "write_dataset",
]
