"""The *prepared benchmark bundle* format (``BENCHMARK_DATA_FORMAT.md`` §1).

One directory per benchmark — ``benchmark.yaml``, ``table.parquet``, optionally
``structures.extxyz``, ``provenance.yaml`` — written by a ``remedi-data``
preparer and read by this package. This subpackage owns the schema
(:mod:`spec`, :mod:`provenance`), the identity assignment (:mod:`identity`),
the invariants (:mod:`validate`), the disk io (:mod:`bundle`) and the
``smiles`` -> ``conformers`` row expansion (:mod:`stage_transition`).

It is deliberately torch-free.
"""

from remedi.data_handling.bundle.bundle import (
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    STRUCTURES_FILENAME,
    TABLE_FILENAME,
    Bundle,
    BundleValidationError,
    content_hash_of_table,
    discover_bundles,
    normalize_table,
    read_bundle,
    read_table,
    sha256_of_file,
    write_bundle,
)
from remedi.data_handling.bundle.identity import (
    CanonicalSmilesPair,
    IdentityTable,
    SmilesParseError,
    assign_identity,
    canonical_smiles_pair,
    mirror_isomeric_smiles,
)
from remedi.data_handling.bundle.provenance import (
    BundleCounts,
    BundleOutputs,
    BundleProvenance,
    ConformerGenerationRecord,
    EtkdgParameters,
    FileHash,
    GeometryLimits,
    MmffParameters,
    PreparerRecord,
    SourceRecord,
    StructuresOutputRecord,
    TableOutputRecord,
)
from remedi.data_handling.bundle.spec import (
    FIXED_COLUMNS,
    SPLIT_VALUES,
    BenchmarkSpec,
    BenchmarkTask,
    BundleStage,
    EvalMetric,
    GeometryOrigin,
    SplitGroup,
)
from remedi.data_handling.bundle.stage_transition import (
    CONFORMER_EMBEDDING_FAILED,
    ENANTIOMER_PARTNER_FAILED,
    ExpandedBundle,
    expand_to_conformers,
)
from remedi.data_handling.bundle.validate import (
    count_stereoisomer_straddling_constitutions,
    minimum_interatomic_distance,
    stereochemistry_from_frame,
    validate_bundle,
)

__all__ = [
    "CONFORMER_EMBEDDING_FAILED",
    "ENANTIOMER_PARTNER_FAILED",
    "FIXED_COLUMNS",
    "PROVENANCE_FILENAME",
    "SPEC_FILENAME",
    "SPLIT_VALUES",
    "STRUCTURES_FILENAME",
    "TABLE_FILENAME",
    "BenchmarkSpec",
    "BenchmarkTask",
    "Bundle",
    "BundleCounts",
    "BundleOutputs",
    "BundleProvenance",
    "BundleStage",
    "BundleValidationError",
    "CanonicalSmilesPair",
    "ConformerGenerationRecord",
    "EtkdgParameters",
    "EvalMetric",
    "ExpandedBundle",
    "FileHash",
    "GeometryLimits",
    "GeometryOrigin",
    "IdentityTable",
    "MmffParameters",
    "PreparerRecord",
    "SmilesParseError",
    "SourceRecord",
    "SplitGroup",
    "StructuresOutputRecord",
    "TableOutputRecord",
    "assign_identity",
    "canonical_smiles_pair",
    "content_hash_of_table",
    "count_stereoisomer_straddling_constitutions",
    "discover_bundles",
    "expand_to_conformers",
    "minimum_interatomic_distance",
    "mirror_isomeric_smiles",
    "normalize_table",
    "read_bundle",
    "read_table",
    "sha256_of_file",
    "stereochemistry_from_frame",
    "validate_bundle",
    "write_bundle",
]
