"""Datasets: structures in a zarr with the table files beside it (§4.1, §6).

:func:`write_dataset` is the one writer every structure dataset goes through;
:func:`build_datasets` turns SMILES bundles into datasets by generating
conformers, then verifies every dataset; :func:`verify_dataset` re-checks one.
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
from remedi_data.build.verify import VerifyReport, verify_dataset
from remedi_data.build.write import (
    ZarrLayout,
    read_packed_structures,
    structures_sha256,
    write_dataset,
    zarr_problems,
)

__all__ = [
    "CONFORMER_EMBEDDING_FAILED",
    "DEFAULT_GEOMETRY_LIMITS",
    "MMFF_NOT_APPLIED",
    "STEREO_MISMATCH_AFTER_EMBEDDING",
    "TIMINGS_FILENAME",
    "BuildDatasetTask",
    "BuildSummary",
    "DatasetBuildConfig",
    "VerifyDatasetTask",
    "VerifyReport",
    "ZarrLayout",
    "build_datasets",
    "embed_bundle",
    "read_packed_structures",
    "structures_sha256",
    "verify_dataset",
    "write_dataset",
    "zarr_problems",
]
