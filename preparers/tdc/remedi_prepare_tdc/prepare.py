"""PyTDC ``admet_group`` -> one bundle per endpoint.

``BENCHMARK_DATA_FORMAT.md`` §5c and §10. For each of the 22 ADMET group
endpoints (listed in ``endpoints.yaml`` next to this module) this reads the
downloaded ``train_val.csv`` + ``test.csv``, cleans the SMILES with the shared
SMILES filter (:func:`remedi.data_handling.chemistry.smiles_filter.filter_smiles`),
merges replicate measurements, assigns identity, freezes the five seeded
train/valid partitions as ``split__seed{1..5}`` alongside the fixed ``test``
fold, and writes ``benchmark_data/bundles/<dataset_id>/``.

Run from the repository root::

    uv run --no-sync prepare-tdc [--only HIA_Hou ...]

The raw download must already be present: ``admet_group`` re-downloads when the
files are missing and this script is meant to be reproducible offline.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import rdBase

from remedi.data_handling.bundle import (
    METRICS_BY_TASK_TYPE,
    Bundle,
    BundleCounts,
    BundleProvenance,
    BundleValidationError,
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    FileHash,
    IdentityTable,
    LabelColumn,
    PreparerRecord,
    SourceRecord,
    assign_identity,
    canonical_smiles_pair,
    metrics_with_headline,
    sha256_of_file,
    write_bundle,
)
from remedi.data_handling.chemistry.smiles_filter import (
    SmilesFilterConfig,
    filter_smiles,
)
from remedi.data_handling.dataset.tasks import TaskType

logger = logging.getLogger("preparers.tdc")

PREPARER_REPO = "molsuit/Rem3Di"
#: The ``Rem3Di`` checkout this module lives in; the default data paths hang off it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "tdc"
DEFAULT_BUNDLE_ROOT = REPOSITORY_ROOT / "benchmark_data" / "bundles"
ENDPOINTS_FILE = Path(__file__).with_name("endpoints.yaml")
SOURCE_KIND = "tdc_admet_group"

#: The column names PyTDC uses in every ``admet_group`` csv.
SMILES_COLUMN = "Drug"
LABEL_COLUMN = "Y"
IDENTIFIER_COLUMN = "Drug_ID"
SOURCE_FILENAMES = ("train_val.csv", "test.csv")

#: The one ``extra_columns`` entry every TDC bundle carries: how many source
#: rows went into this row's label (1 for a compound measured once).
MEASUREMENT_COUNT_COLUMN = "n_measurements"

#: ``counts.dropped`` key for a compound whose classification votes tie exactly,
#: which leaves its label unknown.
DUPLICATE_LABEL_TIE = "duplicate_label_tie"
#: ``counts.dropped`` key for the replicate rows merged into an earlier row.
DUPLICATE_SMILES = "duplicate_smiles"

#: Recorded in ``provenance.yaml`` under ``notices`` on every TDC bundle.
AGGREGATION_NOTICE = (
    "Replicate rows sharing a canonical isomeric SMILES are merged into one row: "
    "the label is the mean of the replicate measurements for a regression "
    "endpoint and their majority vote for a classification endpoint, a compound "
    "whose classification votes tie exactly is dropped as having no known label "
    f"(counts.dropped.{DUPLICATE_LABEL_TIE}), and the {MEASUREMENT_COUNT_COLUMN} "
    "column records how many source rows each label was aggregated from."
)

#: ``tdc.metadata.admet_metrics`` values -> the bundle's metric enum.
TDC_METRIC_TO_EVAL_METRIC: dict[str, EvalMetric] = {
    "mae": EvalMetric.mae,
    "spearman": EvalMetric.spearman,
    "roc-auc": EvalMetric.auroc,
    "pr-auc": EvalMetric.auprc,
}


class TdcPreparationError(RuntimeError):
    """Raised when an endpoint cannot be turned into a valid bundle."""


# ------------------------------------------------------- the endpoint table


class TdcEndpoint(BaseModel):
    """One ``admet_group`` benchmark and the names it takes in a bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The PyTDC benchmark name as the registry spells it; the bundle directory.
    dataset_id: str = Field(min_length=1)
    #: The registry's short label; the label column of ``table.parquet``.
    label_name: str = Field(min_length=1)
    property_description: str = Field(min_length=1)


class TdcEndpointCatalog(BaseModel):
    """``endpoints.yaml``: every endpoint this preparer knows, in run order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoints: list[TdcEndpoint] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_names(self) -> TdcEndpointCatalog:
        for field_name in ("dataset_id", "label_name"):
            values = [getattr(endpoint, field_name) for endpoint in self.endpoints]
            duplicates = sorted({value for value in values if values.count(value) > 1})
            if duplicates:
                raise ValueError(f"duplicate {field_name} values {duplicates}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> TdcEndpointCatalog:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def dataset_ids(self) -> list[str]:
        return [endpoint.dataset_id for endpoint in self.endpoints]


#: All 22 endpoints of the TDC ADMET benchmark group (§5c).
TDC_ENDPOINTS: tuple[TdcEndpoint, ...] = tuple(
    TdcEndpointCatalog.from_yaml(ENDPOINTS_FILE).endpoints
)


# -------------------------------------------------------------- the config


class TdcPreparerConfig(BaseModel):
    """Knobs of one preparer run."""

    model_config = ConfigDict(extra="forbid")

    #: Directory handed to ``admet_group(path=...)``; holds ``admet_group/``.
    raw_root: Path = DEFAULT_RAW_ROOT
    #: Directory that receives one ``<dataset_id>/`` bundle per endpoint.
    bundle_root: Path = DEFAULT_BUNDLE_ROOT
    #: The ``get_train_valid_split`` seeds frozen into the bundle. The first one
    #: is what the bare ``split`` column aliases (§5c).
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5], min_length=1)
    #: Recorded verbatim in ``provenance.yaml``.
    smiles_filter: SmilesFilterConfig = Field(default_factory=SmilesFilterConfig)
    #: Restrict the run to these dataset ids; empty means all 22.
    only: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_selection(self) -> TdcPreparerConfig:
        known = {endpoint.dataset_id for endpoint in TDC_ENDPOINTS}
        unknown = sorted(set(self.only) - known)
        if unknown:
            raise ValueError(f"unknown dataset ids {unknown}; known: {sorted(known)}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"duplicate seeds in {self.seeds}")
        if not self.smiles_filter.dedupe:
            raise ValueError(
                "smiles_filter.dedupe must stay on: the preparer collapses "
                "duplicate SMILES itself so that it can aggregate their labels, "
                "and a bundle row is one stereoisomer"
            )
        return self

    def selected_endpoints(self) -> list[TdcEndpoint]:
        if not self.only:
            return list(TDC_ENDPOINTS)
        wanted = set(self.only)
        return [endpoint for endpoint in TDC_ENDPOINTS if endpoint.dataset_id in wanted]

    def split_columns(self) -> list[str]:
        """``[split, split__seed1, …]`` — the bare name first, then one per seed."""
        return ["split", *(f"split__seed{seed}" for seed in self.seeds)]


# ------------------------------------------------------------- PyTDC quirks


def patch_tdc_print_sys() -> None:
    """Inject ``print_sys`` into ``tdc.utils.split``.

    PyTDC 0.x (incl. 0.3.6) catches RDKit RuntimeErrors from
    ``MurckoScaffoldSmiles`` inside ``create_scaffold_split`` and tries to log
    a skip message via ``print_sys(...)`` — but ``print_sys`` is never
    imported into ``tdc.utils.split`` so the handler itself raises
    ``NameError``, aborting the whole build (seen on CYP2C9_Veith with
    a 'bad bond stereo' SMILES). Provide the symbol so the intended
    skip-and-continue runs.
    """
    import tdc.utils.split as tdc_split_module

    if hasattr(tdc_split_module, "print_sys"):
        return
    try:
        from tdc.utils import print_sys
    except ImportError:

        def print_sys(message: str, *args: object, **kwargs: object) -> None:
            logger.warning("tdc skipped: %s", message)

    # setattr keeps ty happy — print_sys is intentionally not declared on
    # tdc.utils.split (the bug we're patching).
    setattr(tdc_split_module, "print_sys", print_sys)  # noqa: B010


def _resolve_tdc_directory_name(group: Any, dataset_id: str) -> str:
    """The lowercase directory PyTDC stores an endpoint under."""
    from tdc.utils import fuzzy_search

    return str(fuzzy_search(dataset_id, group.dataset_names))


def load_admet_group(raw_root: Path) -> Any:
    """Open the pinned ``admet_group`` download without touching the network."""
    marker = raw_root / "admet_group"
    if not marker.is_dir():
        raise TdcPreparationError(
            f"{marker} does not exist; the admet_group download must be in place "
            "(this script never downloads)"
        )
    patch_tdc_print_sys()
    from tdc.benchmark_group import admet_group

    return admet_group(path=str(raw_root))


def tdc_default_metrics() -> dict[str, str]:
    """``tdc.metadata.admet_metrics``: lowercase endpoint name -> metric name."""
    from tdc.metadata import admet_metrics

    return dict(admet_metrics)


def package_versions() -> dict[str, str]:
    """``PyTDC`` and ``rdkit`` versions, pinned into every bundle (§5c)."""
    try:
        pytdc_version = installed_version("PyTDC")
    except PackageNotFoundError as error:  # pragma: no cover - PyTDC is a hard dep
        raise TdcPreparationError("PyTDC is not installed") from error
    return {"PyTDC": pytdc_version, "rdkit": rdBase.rdkitVersion}


# ------------------------------------------------------------------ metrics


def headline_metric(tdc_metric: str) -> EvalMetric:
    """The TDC default metric of one endpoint as an :class:`EvalMetric`."""
    try:
        return TDC_METRIC_TO_EVAL_METRIC[tdc_metric]
    except KeyError as error:
        raise TdcPreparationError(
            f"TDC metric {tdc_metric!r} has no EvalMetric equivalent; known: "
            f"{sorted(TDC_METRIC_TO_EVAL_METRIC)}"
        ) from error


def task_type_for_metric(metric: EvalMetric) -> TaskType:
    """The task type whose metric family holds ``metric``."""
    for task_type, family in METRICS_BY_TASK_TYPE.items():
        if metric in family:
            return task_type
    raise TdcPreparationError(f"metric {metric.value} belongs to no task type")


# ------------------------------------------------------------ source tables


@dataclass(frozen=True)
class SourceTable:
    """The raw rows of one endpoint in merge order: ``train_val`` then ``test``.

    The order is the dedupe priority: a SMILES seen in ``train_val`` keeps its
    ``train_val`` label and split when it reappears in ``test``.
    """

    train_val: pd.DataFrame
    test: pd.DataFrame

    def __len__(self) -> int:
        return len(self.train_val) + len(self.test)

    def raw_smiles(self) -> list[str]:
        return [
            *self.train_val[SMILES_COLUMN].tolist(),
            *self.test[SMILES_COLUMN].tolist(),
        ]

    def labels(self) -> list[float]:
        return [
            float(value)
            for value in (
                *self.train_val[LABEL_COLUMN].tolist(),
                *self.test[LABEL_COLUMN].tolist(),
            )
        ]


def read_source_table(raw_root: Path, tdc_directory_name: str) -> SourceTable:
    """Read the two csvs of one endpoint from the pinned download."""
    endpoint_directory = raw_root / "admet_group" / tdc_directory_name
    frames: list[pd.DataFrame] = []
    for filename in SOURCE_FILENAMES:
        path = endpoint_directory / filename
        if not path.is_file():
            raise TdcPreparationError(f"missing raw file {path}")
        frames.append(pd.read_csv(path))
    return SourceTable(train_val=frames[0], test=frames[1])


def source_file_hashes(raw_root: Path, tdc_directory_name: str) -> dict[str, FileHash]:
    """sha256 of the two csvs, keyed relative to ``raw_root``."""
    return {
        f"admet_group/{tdc_directory_name}/{filename}": FileHash(
            sha256=sha256_of_file(
                raw_root / "admet_group" / tdc_directory_name / filename
            )
        )
        for filename in SOURCE_FILENAMES
    }


# ------------------------------------------------------------ split columns


def _content_keys(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    """One hashable key per row over the three columns PyTDC round-trips."""
    columns = [IDENTIFIER_COLUMN, SMILES_COLUMN, LABEL_COLUMN]
    return [tuple(row) for row in frame[columns].itertuples(index=False, name=None)]


def train_valid_labels(
    train_val: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame
) -> list[str]:
    """Map a seeded train/valid partition back onto ``train_val`` row positions.

    ``BenchmarkGroup.get_train_valid_split`` returns re-indexed *copies* of the
    ``train_val`` rows, so the partition has to be matched back by content.
    Rows that agree in every column are interchangeable; when such a group is
    split across train and valid the earlier positions are labelled ``train``.

    Raises:
        TdcPreparationError: if a returned row is not in ``train_val`` or if a
            ``train_val`` row is in neither partition (PyTDC silently drops rows
            whose Murcko scaffold RDKit refuses to compute).
    """
    positions_by_key: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for position, key in enumerate(_content_keys(train_val)):
        positions_by_key[key].append(position)

    labels: list[str | None] = [None] * len(train_val)
    for split_name, frame in (("train", train), ("valid", valid)):
        for key in _content_keys(frame):
            positions = positions_by_key.get(key)
            if not positions:
                raise TdcPreparationError(
                    f"row {key!r} of the {split_name} partition is not an "
                    "unclaimed train_val row"
                )
            labels[positions.pop(0)] = split_name

    unassigned = [position for position, label in enumerate(labels) if label is None]
    if unassigned:
        raise TdcPreparationError(
            f"{len(unassigned)} train_val rows are in neither the train nor the "
            f"valid partition (first positions {unassigned[:5]})"
        )
    return [label for label in labels if label is not None]


def split_column_values(
    source: SourceTable, train: pd.DataFrame, valid: pd.DataFrame
) -> list[str]:
    """The split value of every raw row, in ``train_val`` then ``test`` order."""
    return [
        *train_valid_labels(source.train_val, train, valid),
        *(["test"] * len(source.test)),
    ]


def raw_split_columns(
    group: Any, dataset_id: str, source: SourceTable, seeds: Sequence[int]
) -> dict[str, list[str]]:
    """Every split column over the raw rows; ``split`` aliases the first seed (§5c)."""
    columns: dict[str, list[str]] = {}
    for seed in seeds:
        train, valid = group.get_train_valid_split(
            seed=seed, benchmark=dataset_id, split_type="default"
        )
        columns[f"split__seed{seed}"] = split_column_values(source, train, valid)
    return {"split": list(columns[f"split__seed{seeds[0]}"]), **columns}


# --------------------------------------------- filtering, aggregation, ids


@dataclass(frozen=True)
class FilterOutcome:
    """Every source row :func:`filter_smiles` kept, and why it dropped the rest."""

    kept_row_indices: list[int]
    isomeric_smiles: list[str]
    dropped: dict[str, int]


def filter_source_smiles(
    raw_smiles: Sequence[str], smiles_filter: SmilesFilterConfig
) -> FilterOutcome:
    """Run the shared SMILES filter over the raw SMILES, keeping every occurrence.

    The filter runs with ``dedupe`` off on purpose: the preparer has to see
    *all* measurements of a compound to aggregate them
    (:func:`aggregate_by_canonical_smiles`), so the collapse happens one step
    later rather than inside the filter. ``smiles_filter.dedupe`` stays ``True``
    in the recorded settings because duplicates are still collapsed.

    The filter reports a single ``filtered`` verdict, so the element gate, the
    atom-count gate, the fragment gate and the isotope gate are counted together.
    """
    filtered = filter_smiles(
        raw_smiles, smiles_filter.model_copy(update={"dedupe": False})
    )
    return FilterOutcome(
        kept_row_indices=filtered.kept_row_indices,
        isomeric_smiles=filtered.isomeric_smiles,
        dropped={
            "invalid_smiles": filtered.invalid,
            "smiles_filter": filtered.filtered,
        },
    )


def aggregate_label(values: Sequence[float], task_type: TaskType) -> float | None:
    """Collapse the replicate measurements of one compound into one label.

    A regression endpoint takes the mean of the replicates; a classification
    endpoint takes their majority vote. An exact tie leaves the class unknown,
    so the compound has no label and :func:`aggregate_by_canonical_smiles` drops
    it. A compound measured once keeps its measurement bit for bit.

    Raises:
        TdcPreparationError: if ``values`` is empty, or a classification value
            is neither 0 nor 1. The vote would silently turn such a value into
            a class, so the 0/1 invariant ``write_bundle`` checks could no
            longer see it.
    """
    if not values:
        raise TdcPreparationError("cannot aggregate an empty list of measurements")
    if task_type is not TaskType.classification:
        return statistics.fmean(values)
    invalid = sorted({value for value in values if value not in (0.0, 1.0)})
    if invalid:
        raise TdcPreparationError(
            f"classification measurements {invalid} are neither 0 nor 1"
        )
    positive_votes = sum(1 for value in values if value == 1.0)
    negative_votes = len(values) - positive_votes
    if positive_votes == negative_votes:
        return None
    return 1.0 if positive_votes > negative_votes else 0.0


@dataclass(frozen=True)
class AggregatedRows:
    """One row per stereoisomer, with the replicate measurements collapsed."""

    #: Positions into the :class:`FilterOutcome` lists — the *first* occurrence
    #: of each compound, which is what the split columns are read from.
    first_positions: list[int]
    identity: IdentityTable
    labels: list[float]
    measurement_counts: list[int]
    #: Replicate rows merged into an earlier row (``duplicate_smiles``).
    merged_rows: int
    #: Compounds dropped because their classification votes tied exactly.
    label_ties: int

    @property
    def rows_with_replicates(self) -> int:
        """How many rows carry more than one measurement."""
        return sum(1 for count in self.measurement_counts if count > 1)


def aggregate_by_canonical_smiles(
    isomeric_smiles: Sequence[str],
    labels: Sequence[float],
    task_type: TaskType,
) -> AggregatedRows:
    """Group the filtered rows by canonical isomeric SMILES and merge their labels.

    Groups are emitted in first-occurrence order over the input, which is
    ``train_val`` rows then ``test`` rows, so a compound's identity, split and
    row position all come from where it was first seen.
    """
    canonical = [canonical_smiles_pair(smiles).isomeric for smiles in isomeric_smiles]
    positions_by_smiles: dict[str, list[int]] = {}
    for position, smiles in enumerate(canonical):
        positions_by_smiles.setdefault(smiles, []).append(position)

    first_positions: list[int] = []
    aggregated_labels: list[float] = []
    measurement_counts: list[int] = []
    label_ties = 0
    for positions in positions_by_smiles.values():
        label = aggregate_label([labels[position] for position in positions], task_type)
        if label is None:
            label_ties += 1
            continue
        first_positions.append(positions[0])
        aggregated_labels.append(label)
        measurement_counts.append(len(positions))

    return AggregatedRows(
        first_positions=first_positions,
        identity=assign_identity([canonical[position] for position in first_positions]),
        labels=aggregated_labels,
        measurement_counts=measurement_counts,
        merged_rows=len(canonical) - len(positions_by_smiles),
        label_ties=label_ties,
    )


def check_row_accounting(
    source_molecules: int, final_rows: int, dropped: dict[str, int]
) -> None:
    """Every source row must be either a bundle row or counted under one reason.

    Raises:
        TdcPreparationError: if the two sides do not add up.
    """
    accounted = final_rows + sum(dropped.values())
    if accounted != source_molecules:
        raise TdcPreparationError(
            f"{final_rows} rows + dropped {dropped} = {accounted}, which does "
            f"not account for the {source_molecules} source rows"
        )


# ------------------------------------------------------------- table + spec


def build_table(
    identity: IdentityTable,
    label_name: str,
    labels: Sequence[float],
    split_values: dict[str, list[str]],
    measurement_counts: Sequence[int],
) -> pd.DataFrame:
    """Identity and SMILES columns, the label, the split columns, ``n_measurements``.

    Raises:
        TdcPreparationError: on a missing label, an ``unassigned`` split value or
            a non-positive measurement count — none can happen on this source,
            the format allows the first two, and all would be silent corruption
            downstream.
    """
    table = identity.to_frame()
    table[label_name] = pd.Series(list(labels), dtype="float64", index=table.index)
    if table[label_name].isna().any():
        raise TdcPreparationError(
            f"label column {label_name!r} has NaN labels; TDC always supplies one"
        )
    for column_name, values in split_values.items():
        table[column_name] = pd.Series(list(values), dtype="str", index=table.index)
        if (table[column_name] == "unassigned").any():
            raise TdcPreparationError(
                f"split column {column_name!r} has unassigned rows"
            )
    table[MEASUREMENT_COUNT_COLUMN] = pd.Series(
        list(measurement_counts), dtype="float64", index=table.index
    )
    if (table[MEASUREMENT_COUNT_COLUMN] < 1).any():
        raise TdcPreparationError(
            f"{MEASUREMENT_COUNT_COLUMN} must be at least 1 on every row"
        )
    return table


def build_spec(
    endpoint: TdcEndpoint, tdc_metric: str, split_columns: Sequence[str]
) -> DatasetSpec:
    """``dataset.yaml`` for one endpoint."""
    headline = headline_metric(tdc_metric)
    task_type = task_type_for_metric(headline)
    return DatasetSpec(
        dataset_id=endpoint.dataset_id,
        description=(
            f"TDC ADMET benchmark group endpoint {endpoint.dataset_id} "
            f"({endpoint.property_description}), from tdcommons.ai. Rows are the "
            "union of the official train_val and test folds after the Rem3Di "
            "SMILES filter, one row per stereoisomer with replicate "
            "measurements aggregated; the fixed TDC test fold plus five seeded "
            "train/valid partitions are frozen as the split columns."
        ),
        smiles=True,
        labels=[LabelColumn(name=endpoint.label_name, task_type=task_type)],
        extra_columns=[MEASUREMENT_COUNT_COLUMN],
        evaluation=EvaluationSpec(
            metrics=metrics_with_headline(headline, task_type),
            split_columns=list(split_columns),
            default_split="split",
            split_group="stereoisomer_id",
            require_enantiomer_pairs=False,
        ),
        source_kind=SOURCE_KIND,
    )


# ------------------------------------------------------------- the endpoint


@dataclass(frozen=True)
class EndpointReport:
    """What one prepared endpoint came out as, for the run summary."""

    dataset_id: str
    label_name: str
    headline_metric: EvalMetric
    task_type: TaskType
    #: How many bundle rows carry more than one measurement.
    rows_with_replicates: int
    #: The provenance as written, counts and output hashes filled in.
    provenance: BundleProvenance
    directory: Path

    @property
    def counts(self) -> BundleCounts:
        return self.provenance.counts

    @property
    def content_sha256(self) -> str:
        if self.provenance.outputs is None:  # pragma: no cover - the writer fills it
            raise TdcPreparationError(f"{self.directory} has no output hashes")
        return self.provenance.outputs.table_parquet.content_sha256


def prepare_endpoint(
    endpoint: TdcEndpoint,
    config: TdcPreparerConfig,
    group: Any,
    tdc_metric: str,
    preparer: PreparerRecord,
) -> EndpointReport:
    """Read, clean, split, aggregate and write one endpoint.

    Raises:
        TdcPreparationError: on a source problem this preparer detects.
        BundleValidationError: if the table violates a format invariant.
    """
    tdc_directory_name = _resolve_tdc_directory_name(group, endpoint.dataset_id)
    source = read_source_table(config.raw_root, tdc_directory_name)
    spec = build_spec(endpoint, tdc_metric, config.split_columns())
    task_type = spec.labels[0].task_type
    split_columns = raw_split_columns(group, endpoint.dataset_id, source, config.seeds)

    filtered = filter_source_smiles(source.raw_smiles(), config.smiles_filter)
    raw_labels = source.labels()
    aggregated = aggregate_by_canonical_smiles(
        filtered.isomeric_smiles,
        [raw_labels[index] for index in filtered.kept_row_indices],
        task_type,
    )
    kept_raw_indices = [
        filtered.kept_row_indices[position] for position in aggregated.first_positions
    ]

    dropped = {
        **filtered.dropped,
        DUPLICATE_SMILES: aggregated.merged_rows,
        DUPLICATE_LABEL_TIE: aggregated.label_ties,
    }
    check_row_accounting(len(source), len(kept_raw_indices), dropped)

    table = build_table(
        identity=aggregated.identity,
        label_name=endpoint.label_name,
        labels=aggregated.labels,
        split_values={
            column_name: [
                split_columns[column_name][index] for index in kept_raw_indices
            ]
            for column_name in spec.split_columns()
        },
        measurement_counts=aggregated.measurement_counts,
    )
    provenance = BundleProvenance(
        dataset_id=endpoint.dataset_id,
        preparer=preparer,
        source=SourceRecord(
            files=source_file_hashes(config.raw_root, tdc_directory_name),
            package_versions=package_versions(),
        ),
        smiles_filter=config.smiles_filter,
        counts=BundleCounts(source_molecules=len(source), dropped=dropped),
        notices=[AGGREGATION_NOTICE],
    )

    directory = config.bundle_root / endpoint.dataset_id
    written = write_bundle(
        Bundle(spec=spec, table=table, provenance=provenance), directory
    )
    return EndpointReport(
        dataset_id=endpoint.dataset_id,
        label_name=endpoint.label_name,
        headline_metric=headline_metric(tdc_metric),
        task_type=task_type,
        rows_with_replicates=aggregated.rows_with_replicates,
        provenance=written,
        directory=directory,
    )


# --------------------------------------------------------------------- main


def format_report(reports: Iterable[EndpointReport]) -> str:
    """A fixed-width summary of a run, one line per endpoint."""
    header = (
        f"{'dataset_id':32s} {'label':16s} {'metric':9s} {'source':>7s} "
        f"{'final':>7s} {'inval':>6s} {'filt':>6s} {'dup':>6s} {'tie':>4s} "
        f"{'aggr':>5s} {'train':>7s} {'valid':>6s} {'test':>6s} {'straddle':>8s} "
        f"{'content_sha256':14s}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        counts = report.counts
        lines.append(
            f"{report.dataset_id:32s} {report.label_name:16s} "
            f"{report.headline_metric.value:9s} {counts.source_molecules:7d} "
            f"{counts.final_rows:7d} {counts.dropped.get('invalid_smiles', 0):6d} "
            f"{counts.dropped.get('smiles_filter', 0):6d} "
            f"{counts.dropped.get(DUPLICATE_SMILES, 0):6d} "
            f"{counts.dropped.get(DUPLICATE_LABEL_TIE, 0):4d} "
            f"{report.rows_with_replicates:5d} "
            f"{counts.per_split.get('train', 0):7d} "
            f"{counts.per_split.get('valid', 0):6d} "
            f"{counts.per_split.get('test', 0):6d} "
            f"{counts.stereoisomer_straddling_constitutions:8d} "
            f"{report.content_sha256[:14]:14s}"
        )
    return "\n".join(lines)


def run(config: TdcPreparerConfig) -> list[EndpointReport]:
    """Prepare every selected endpoint. Returns one report each."""
    group = load_admet_group(config.raw_root)
    metrics = tdc_default_metrics()
    preparer = PreparerRecord.for_script(PREPARER_REPO, Path(__file__))
    reports: list[EndpointReport] = []
    for endpoint in config.selected_endpoints():
        directory_name = _resolve_tdc_directory_name(group, endpoint.dataset_id)
        if directory_name not in metrics:
            raise TdcPreparationError(
                f"{endpoint.dataset_id} resolves to {directory_name!r}, which is "
                "not in tdc.metadata.admet_metrics"
            )
        logger.info("preparing %s", endpoint.dataset_id)
        reports.append(
            prepare_endpoint(
                endpoint=endpoint,
                config=config,
                group=group,
                tdc_metric=metrics[directory_name],
                preparer=preparer,
            )
        )
    return reports


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="DATASET_ID",
        help="prepare only this endpoint; repeatable",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="the get_train_valid_split seeds to freeze",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    arguments = parse_arguments(argv)
    config = TdcPreparerConfig(
        raw_root=arguments.raw_root,
        bundle_root=arguments.bundle_root,
        seeds=list(arguments.seeds),
        only=list(arguments.only),
    )
    try:
        reports = run(config)
    except (TdcPreparationError, BundleValidationError) as error:
        logger.error("%s", error)
        return 1
    print(format_report(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
