"""Datasets: structures in a zarr with the table files beside it (§10.4).

:func:`write_dataset` is the one writer every structure dataset goes through;
:func:`build_datasets` turns SMILES bundles into datasets by generating
conformers, then verifies every dataset; :func:`verify_dataset` re-checks one.
"""

from remedi.data_handling.dataset_build.build import (
    DEFAULT_GEOMETRY_LIMITS,
    STEREO_MISMATCH_AFTER_EMBEDDING,
    TIMINGS_FILENAME,
    BuildDatasetTask,
    BuildSummary,
    DatasetBuildConfig,
    VerifyDatasetTask,
    build_datasets,
    embed_bundle,
)
from remedi.data_handling.dataset_build.verify import VerifyReport, verify_dataset
from remedi.data_handling.dataset_build.write import (
    ZarrLayout,
    structures_sha256,
    write_dataset,
    zarr_problems,
)

__all__ = [
    "DEFAULT_GEOMETRY_LIMITS",
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
    "structures_sha256",
    "verify_dataset",
    "write_dataset",
    "zarr_problems",
]
