"""The toolkit of the SMILES preparers in ``preparers/`` (``BENCHMARK_DATA_FORMAT.md`` §5.1).

The source-row merge every SMILES preparer shares (:mod:`aggregation`), the
pinned source files (:mod:`pinned_files`) and the bundle a preparer writes
from its merged rows, with its command line (:mod:`preparation`).
"""

from remedi_data.preparation.smiles.aggregation import (
    AGGREGATION_NOTICE,
    DUPLICATE_LABEL_TIE,
    DUPLICATE_SMILES,
    INVALID_SMILES,
    MEASUREMENT_COUNT_COLUMN,
    MISSING_LABEL,
    SMILES_FILTER,
    AggregatedRows,
    FilterOutcome,
    LabelAggregationError,
    MergedRows,
    aggregate_by_canonical_smiles,
    aggregate_label,
    check_row_accounting,
    filter_source_smiles,
    merge_source_rows,
)
from remedi_data.preparation.smiles.pinned_files import (
    PinnedFile,
    PinnedFileError,
    ensure_pinned_file,
)
from remedi_data.preparation.smiles.preparation import (
    PREPARER_REPOSITORY,
    BundleReport,
    PreparationError,
    PreparerSettings,
    SourceLabel,
    check_unique,
    format_report,
    preparer_main,
    seeded_split_notice,
    smiles_bundle_spec,
    smiles_bundle_table,
    smiles_evaluation_metrics,
    write_smiles_bundle,
)

__all__ = [
    "AGGREGATION_NOTICE",
    "DUPLICATE_LABEL_TIE",
    "DUPLICATE_SMILES",
    "INVALID_SMILES",
    "MEASUREMENT_COUNT_COLUMN",
    "MISSING_LABEL",
    "PREPARER_REPOSITORY",
    "SMILES_FILTER",
    "AggregatedRows",
    "BundleReport",
    "FilterOutcome",
    "LabelAggregationError",
    "MergedRows",
    "PinnedFile",
    "PinnedFileError",
    "PreparationError",
    "PreparerSettings",
    "SourceLabel",
    "aggregate_by_canonical_smiles",
    "aggregate_label",
    "check_row_accounting",
    "check_unique",
    "ensure_pinned_file",
    "filter_source_smiles",
    "format_report",
    "merge_source_rows",
    "preparer_main",
    "seeded_split_notice",
    "smiles_bundle_spec",
    "smiles_bundle_table",
    "smiles_evaluation_metrics",
    "write_smiles_bundle",
]
