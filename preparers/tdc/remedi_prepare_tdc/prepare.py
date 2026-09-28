"""PyTDC ``admet_group`` -> one ``smiles``-stage benchmark bundle per endpoint.

``BENCHMARK_DATA_FORMAT.md`` step 2. For each of the 22 ADMET group endpoints
this reads the downloaded ``train_val.csv`` + ``test.csv``, cleans the SMILES
with the same filter the Rem3Di ingest pipeline uses today
(:func:`apply_smiles_filter` with the defaults of
``FilterMoleculeStageConfig``), assigns bundle identity, freezes the five
seeded train/valid partitions as ``split__seed{1..5}`` alongside the fixed
``test`` fold, and writes ``benchmark_data/bundles/<dataset_id>/``.

Run from the repository root::

    uv run --no-sync prepare-tdc [--only HIA_Hou ...]

The raw download must already be present: ``admet_group`` re-downloads when the
files are missing and this script is meant to be reproducible offline.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import rdBase

from remedi.configuration.dataset_config import FilterMoleculeStageConfig
from remedi.data_handling.bundle import (
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    TABLE_FILENAME,
    BenchmarkSpec,
    BenchmarkTask,
    Bundle,
    BundleCounts,
    BundleProvenance,
    EvalMetric,
    FileHash,
    IdentityTable,
    PreparerRecord,
    SmilesFilterRecord,
    SourceRecord,
    assign_identity,
    canonical_smiles_pair,
    read_bundle,
    sha256_of_file,
    write_bundle,
)
from remedi.data_handling.dataset.tasks import TaskType
from remedi.data_handling.dataset_creation.build_stats import LoadStats
from remedi.data_handling.dataset_creation.generators.utils import (
    apply_smiles_filter,
    resolve_element_set,
)

logger = logging.getLogger("preparers.tdc")

#: This module's path inside the ``Rem3Di`` repository, for provenance.
PREPARER_SCRIPT = "preparers/tdc/remedi_prepare_tdc/prepare.py"
PREPARER_REPO = "molsuit/Rem3Di"
#: The ``Rem3Di`` checkout this module lives in; the default data paths hang off it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "tdc"
DEFAULT_BUNDLE_ROOT = REPOSITORY_ROOT / "benchmark_data" / "bundles"
SOURCE_KIND = "tdc_admet_group"

#: The column names PyTDC uses in every ``admet_group`` csv.
SMILES_COLUMN = "Drug"
LABEL_COLUMN = "Y"
IDENTIFIER_COLUMN = "Drug_ID"

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

#: Directory name used next to ``bundle_root`` while a bundle is being written.
STAGING_DIRECTORY_NAME = ".preparer-staging"


class TdcPreparationError(RuntimeError):
    """Raised when an endpoint cannot be turned into a valid bundle."""


# ------------------------------------------------------- the endpoint table


class TdcEndpoint(BaseModel):
    """One ``admet_group`` benchmark and the names it takes in a bundle.

    ``dataset_id`` is the PyTDC benchmark name as the Rem3Di registry spells it
    and is also the bundle directory name; ``task_name`` is the registry's short
    label and becomes the scored column of ``table.parquet``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    property_description: str = Field(min_length=1)


#: All 22 endpoints of the TDC ADMET benchmark group (§5c).
TDC_ENDPOINTS: tuple[TdcEndpoint, ...] = (
    TdcEndpoint(
        dataset_id="Caco2_Wang",
        task_name="Caco-2",
        property_description="Caco-2 cell effective permeability",
    ),
    TdcEndpoint(
        dataset_id="HIA_Hou",
        task_name="HIA",
        property_description="human intestinal absorption",
    ),
    TdcEndpoint(
        dataset_id="Pgp_Broccatelli",
        task_name="Pgp",
        property_description="P-glycoprotein inhibition",
    ),
    TdcEndpoint(
        dataset_id="Bioavailability_Ma",
        task_name="Bioavailability",
        property_description="oral bioavailability",
    ),
    TdcEndpoint(
        dataset_id="Lipophilicity_AstraZeneca",
        task_name="Lipophilicity",
        property_description="octanol/water distribution coefficient (logD at pH 7.4)",
    ),
    TdcEndpoint(
        dataset_id="Solubility_AqSolDB",
        task_name="Solubility",
        property_description="aqueous solubility (log mol/L)",
    ),
    TdcEndpoint(
        dataset_id="BBB_Martins",
        task_name="BBB",
        property_description="blood-brain barrier penetration",
    ),
    TdcEndpoint(
        dataset_id="PPBR_AZ",
        task_name="PPBR",
        property_description="human plasma protein binding rate",
    ),
    TdcEndpoint(
        dataset_id="VDss_Lombardo",
        task_name="VDss",
        property_description="volume of distribution at steady state",
    ),
    TdcEndpoint(
        dataset_id="CYP2C9_Veith",
        task_name="CYP2C9-I",
        property_description="CYP2C9 inhibition",
    ),
    TdcEndpoint(
        dataset_id="CYP2D6_Veith",
        task_name="CYP2D6-I",
        property_description="CYP2D6 inhibition",
    ),
    TdcEndpoint(
        dataset_id="CYP3A4_Veith",
        task_name="CYP3A4-I",
        property_description="CYP3A4 inhibition",
    ),
    TdcEndpoint(
        dataset_id="CYP2C9_Substrate_CarbonMangels",
        task_name="CYP2C9-S",
        property_description="CYP2C9 substrate",
    ),
    TdcEndpoint(
        dataset_id="CYP2D6_Substrate_CarbonMangels",
        task_name="CYP2D6-S",
        property_description="CYP2D6 substrate",
    ),
    TdcEndpoint(
        dataset_id="CYP3A4_Substrate_CarbonMangels",
        task_name="CYP3A4-S",
        property_description="CYP3A4 substrate",
    ),
    TdcEndpoint(
        dataset_id="Half_Life_Obach",
        task_name="Half-life",
        property_description="drug half life in the human body",
    ),
    TdcEndpoint(
        dataset_id="Clearance_Hepatocyte_AZ",
        task_name="CL-hepa",
        property_description="drug clearance in human hepatocytes",
    ),
    TdcEndpoint(
        dataset_id="Clearance_Microsome_AZ",
        task_name="CL-micro",
        property_description="drug clearance in human liver microsomes",
    ),
    TdcEndpoint(
        dataset_id="LD50_Zhu",
        task_name="LD50",
        property_description="acute toxicity (median lethal dose)",
    ),
    TdcEndpoint(
        dataset_id="hERG",
        task_name="hERG",
        property_description="hERG channel blocking",
    ),
    TdcEndpoint(
        dataset_id="AMES",
        task_name="AMES",
        property_description="Ames mutagenicity",
    ),
    TdcEndpoint(
        dataset_id="DILI",
        task_name="DILI",
        property_description="drug-induced liver injury",
    ),
)

ENDPOINTS_BY_DATASET_ID: dict[str, TdcEndpoint] = {
    endpoint.dataset_id: endpoint for endpoint in TDC_ENDPOINTS
}


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
    #: Today's ingest defaults; recorded verbatim in ``provenance.yaml``.
    smiles_filter: FilterMoleculeStageConfig = Field(
        default_factory=FilterMoleculeStageConfig
    )
    #: Restrict the run to these dataset ids; empty means all 22.
    only: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_selection(self) -> TdcPreparerConfig:
        unknown = sorted(set(self.only) - set(ENDPOINTS_BY_DATASET_ID))
        if unknown:
            raise ValueError(
                f"unknown dataset ids {unknown}; known: "
                f"{sorted(ENDPOINTS_BY_DATASET_ID)}"
            )
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

    def smiles_filter_record(self) -> SmilesFilterRecord:
        return SmilesFilterRecord(**self.smiles_filter.model_dump(exclude={"kind"}))

    def staging_root(self) -> Path:
        """Where a bundle is assembled before its files are renamed into place.

        Deliberately *outside* ``bundle_root``: ``discover_bundles`` globs for
        ``benchmark.yaml`` recursively, and a concurrent reader must not find a
        half-written bundle.
        """
        return self.bundle_root.parent / STAGING_DIRECTORY_NAME


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
    import tdc.utils.split as _tdc_split

    if hasattr(_tdc_split, "print_sys"):
        return
    try:
        from tdc.utils import print_sys as _print_sys
    except ImportError:

        def _print_sys(msg: str, *args: object, **kwargs: object) -> None:
            logger.warning("tdc skipped: %s", msg)

    # setattr keeps ty happy — print_sys is intentionally not declared on
    # tdc.utils.split (the bug we're patching).
    setattr(_tdc_split, "print_sys", _print_sys)  # noqa: B010


# ------------------------------------------------------- metrics and tasks


#: ``tdc.metadata.admet_metrics`` values -> the bundle's metric enum.
TDC_METRIC_TO_EVAL_METRIC: dict[str, EvalMetric] = {
    "mae": EvalMetric.mae,
    "spearman": EvalMetric.spearman,
    "roc-auc": EvalMetric.auroc,
    "pr-auc": EvalMetric.auprc,
}

#: Metrics computed alongside the headline one (§7e); they are free.
CLASSIFICATION_METRICS: tuple[EvalMetric, ...] = (EvalMetric.auroc, EvalMetric.auprc)
REGRESSION_METRICS: tuple[EvalMetric, ...] = (
    EvalMetric.mae,
    EvalMetric.rmse,
    EvalMetric.spearman,
    EvalMetric.r2,
)


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
    """Classification iff the TDC default metric is a ranking/threshold metric."""
    if metric in CLASSIFICATION_METRICS:
        return TaskType.classification
    return TaskType.regression


def metrics_for_endpoint(tdc_metric: str) -> list[EvalMetric]:
    """The TDC default first, then the rest of its family, de-duplicated (§7e)."""
    headline = headline_metric(tdc_metric)
    task_type = task_type_for_metric(headline)
    family = (
        CLASSIFICATION_METRICS
        if task_type is TaskType.classification
        else REGRESSION_METRICS
    )
    ordered: list[EvalMetric] = []
    for metric in (headline, *family):
        if metric not in ordered:
            ordered.append(metric)
    return ordered


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
    for filename in ("train_val.csv", "test.csv"):
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
        for filename in ("train_val.csv", "test.csv")
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
    split across train and valid the earlier positions are labelled ``train``,
    which reproduces the train-before-valid priority the previous generator had.

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


# --------------------------------------------- filtering, aggregation, ids


@dataclass(frozen=True)
class FilterOutcome:
    """Every source row :func:`apply_smiles_filter` kept, and why it dropped the rest."""

    kept_row_indices: list[int]
    isomeric_smiles: list[str]
    dropped: dict[str, int]


def filter_source_smiles(
    raw_smiles: Sequence[str], smiles_filter: FilterMoleculeStageConfig
) -> FilterOutcome:
    """Run today's ingest filter over the raw SMILES, keeping every occurrence.

    ``apply_smiles_filter`` is called with ``dedupe=False`` on purpose: the
    preparer has to see *all* measurements of a compound to aggregate them
    (:func:`aggregate_by_canonical_smiles`), so the collapse happens one step
    later rather than inside the filter. ``smiles_filter.dedupe`` stays ``True``
    in the recorded settings because duplicates are still collapsed.

    The filter reports a single ``filtered`` verdict, so the element gate, the
    atom-count gate, the fragment gate and the isotope gate cannot be told apart
    here without duplicating its logic; they are counted together.
    """
    stats = LoadStats()
    kept, kept_indices = apply_smiles_filter(
        raw_smiles,
        max_atoms=smiles_filter.max_atoms,
        allowed_elements=resolve_element_set(smiles_filter.element_set),
        allow_charged=smiles_filter.allow_charged,
        allow_radicals=smiles_filter.allow_radicals,
        allow_isotopes=smiles_filter.allow_isotopes,
        allow_multifragment=smiles_filter.allow_multifragment,
        strip_salts=smiles_filter.strip_salts,
        neutralize=smiles_filter.neutralize,
        dedupe=False,
        stats=stats,
    )
    return FilterOutcome(
        kept_row_indices=list(kept_indices),
        isomeric_smiles=[entry.isomeric_smiles for entry in kept],
        dropped={
            "invalid_smiles": stats.n_invalid_smiles,
            "smiles_filter": stats.n_filtered_out,
        },
    )


def aggregate_label(values: Sequence[float], task_type: TaskType) -> float | None:
    """Collapse the replicate measurements of one compound into one label.

    A regression endpoint takes the mean of the replicates; a classification
    endpoint takes their majority vote. An exact tie leaves the class unknown,
    so the compound has no label and :func:`aggregate_by_canonical_smiles` drops
    it. A compound measured once keeps its measurement bit for bit.

    Raises:
        TdcPreparationError: if ``values`` is empty.
    """
    if not values:
        raise TdcPreparationError("cannot aggregate an empty list of measurements")
    if task_type is not TaskType.classification:
        return statistics.fmean(values)
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
        measurements = [labels[position] for position in positions]
        label = aggregate_label(measurements, task_type)
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


# ------------------------------------------------------------- table + spec


def build_table(
    identity: IdentityTable,
    task_name: str,
    labels: Sequence[float],
    split_values: dict[str, list[str]],
    measurement_counts: Sequence[int],
) -> pd.DataFrame:
    """The six fixed columns, the task column, the split columns, ``n_measurements``.

    Raises:
        TdcPreparationError: on a missing label, an ``unassigned`` split value or
            a non-positive measurement count — none can happen on this source
            and all would be silent corruption downstream.
    """
    table = identity.to_frame()
    table[task_name] = pd.Series(list(labels), dtype="float64", index=table.index)
    if table[task_name].isna().any():
        raise TdcPreparationError(
            f"task column {task_name!r} has NaN labels; TDC always supplies one"
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
    endpoint: TdcEndpoint,
    tdc_metric: str,
    split_columns: Sequence[str],
) -> BenchmarkSpec:
    """``benchmark.yaml`` for one endpoint."""
    metrics = metrics_for_endpoint(tdc_metric)
    return BenchmarkSpec(
        dataset_id=endpoint.dataset_id,
        description=(
            f"TDC ADMET benchmark group endpoint {endpoint.dataset_id} "
            f"({endpoint.property_description}), from tdcommons.ai. Rows are the "
            "union of the official train_val and test folds after the Rem3Di "
            "SMILES filter, one row per stereoisomer with replicate "
            "measurements aggregated; the fixed TDC test fold plus five seeded "
            "train/valid partitions are frozen as the split columns."
        ),
        tasks=[
            BenchmarkTask(
                name=endpoint.task_name,
                task_type=task_type_for_metric(metrics[0]),
            )
        ],
        metrics=metrics,
        stage="smiles",
        split_columns=list(split_columns),
        default_split="split",
        split_group="stereoisomer_id",
        require_enantiomer_pairs=False,
        extra_columns=[MEASUREMENT_COUNT_COLUMN],
        source_kind=SOURCE_KIND,
    )


def check_classification_labels(task_name: str, labels: Sequence[float]) -> None:
    """A classification task column must hold exactly the two labels 0 and 1."""
    observed = sorted({float(value) for value in labels})
    if not set(observed) <= {0.0, 1.0}:
        raise TdcPreparationError(
            f"classification task {task_name!r} has labels {observed[:10]}, "
            "expected only 0.0 and 1.0"
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


def write_bundle_atomically(
    bundle: Bundle, directory: Path, staging_root: Path
) -> Path:
    """``write_bundle`` into a staging directory, then rename the files into place.

    Another process reads the bundle root while this runs, so no reader may see
    a half-written ``table.parquet``. The three files are moved one at a time —
    ``os.replace`` is atomic per file within a filesystem — with
    ``provenance.yaml``, which carries the hashes of the other two, last. The
    bundle directory itself is created if missing and never removed; the
    staging directory is, as soon as its files are in place.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    staging_root = Path(staging_root)
    staging = staging_root / directory.name
    shutil.rmtree(staging, ignore_errors=True)
    try:
        write_bundle(bundle, staging)
        for filename in (TABLE_FILENAME, SPEC_FILENAME, PROVENANCE_FILENAME):
            os.replace(staging / filename, directory / filename)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        with contextlib.suppress(OSError):
            staging_root.rmdir()
    return directory


def preparer_record(repository_root: Path) -> PreparerRecord:
    """``preparer:`` — this script and the git sha of the checkout it ran from."""
    return PreparerRecord(
        repo=PREPARER_REPO,
        script=PREPARER_SCRIPT,
        git_sha=git_head_sha(repository_root),
    )


def git_head_sha(repository_root: Path) -> str | None:
    """``git rev-parse HEAD`` in ``repository_root``, or ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("could not read the git sha of %s: %s", repository_root, error)
        return None
    return completed.stdout.strip() or None


def package_versions() -> dict[str, str]:
    """``PyTDC`` and ``rdkit`` versions, pinned into every bundle (§5c)."""
    try:
        pytdc_version = installed_version("PyTDC")
    except PackageNotFoundError as error:  # pragma: no cover - PyTDC is a hard dep
        raise TdcPreparationError("PyTDC is not installed") from error
    return {"PyTDC": pytdc_version, "rdkit": rdBase.rdkitVersion}


# ------------------------------------------------------------- the endpoint


@dataclass(frozen=True)
class EndpointReport:
    """What one prepared endpoint came out as, for the run summary."""

    dataset_id: str
    task_name: str
    headline_metric: EvalMetric
    task_type: TaskType
    source_molecules: int
    final_rows: int
    dropped: dict[str, int]
    per_split: dict[str, int]
    straddling_constitutions: int
    #: How many bundle rows carry more than one measurement.
    aggregated_rows: int
    content_sha256: str
    directory: Path


def prepare_endpoint(
    endpoint: TdcEndpoint,
    config: TdcPreparerConfig,
    group: Any,
    tdc_metric: str,
    preparer: PreparerRecord,
) -> EndpointReport:
    """Read, clean, split and write one endpoint; re-read the bundle to verify."""
    tdc_directory_name = _resolve_tdc_directory_name(group, endpoint.dataset_id)
    source = read_source_table(config.raw_root, tdc_directory_name)
    spec = build_spec(endpoint, tdc_metric, config.split_columns())

    raw_split_values: dict[str, list[str]] = {}
    for seed in config.seeds:
        train, valid = group.get_train_valid_split(
            seed=seed, benchmark=endpoint.dataset_id, split_type="default"
        )
        raw_split_values[f"split__seed{seed}"] = split_column_values(
            source, train, valid
        )
    # `split` aliases the first seed (§5c).
    raw_split_values["split"] = list(raw_split_values[f"split__seed{config.seeds[0]}"])

    filtered = filter_source_smiles(source.raw_smiles(), config.smiles_filter)
    raw_labels = source.labels()
    kept_labels = [raw_labels[index] for index in filtered.kept_row_indices]
    task_type = spec.tasks[0].task_type
    if task_type is TaskType.classification:
        check_classification_labels(endpoint.task_name, kept_labels)

    aggregated = aggregate_by_canonical_smiles(
        filtered.isomeric_smiles, kept_labels, task_type
    )
    kept_raw_indices = [
        filtered.kept_row_indices[position] for position in aggregated.first_positions
    ]

    dropped = dict(filtered.dropped)
    dropped[DUPLICATE_SMILES] = aggregated.merged_rows
    dropped[DUPLICATE_LABEL_TIE] = aggregated.label_ties
    check_row_accounting(len(source), len(kept_raw_indices), dropped)

    table = build_table(
        identity=aggregated.identity,
        task_name=endpoint.task_name,
        labels=aggregated.labels,
        split_values={
            column_name: [
                raw_split_values[column_name][index] for index in kept_raw_indices
            ]
            for column_name in spec.split_columns
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
        smiles_filter=config.smiles_filter_record(),
        counts=BundleCounts(source_molecules=len(source), dropped=dropped),
        notices=[AGGREGATION_NOTICE],
    )

    directory = config.bundle_root / endpoint.dataset_id
    write_bundle_atomically(
        Bundle(spec=spec, table=table, structures=None, provenance=provenance),
        directory,
        config.staging_root(),
    )
    written = read_bundle(directory)
    counts = written.provenance.counts
    if written.provenance.outputs is None:  # pragma: no cover - write_bundle fills it
        raise TdcPreparationError(f"{directory} was written without output hashes")
    return EndpointReport(
        dataset_id=endpoint.dataset_id,
        task_name=endpoint.task_name,
        headline_metric=spec.metrics[0],
        task_type=task_type,
        source_molecules=counts.source_molecules,
        final_rows=counts.final_rows,
        dropped=dict(counts.dropped),
        per_split=dict(counts.per_split),
        straddling_constitutions=counts.stereoisomer_straddling_constitutions,
        aggregated_rows=int((written.table[MEASUREMENT_COUNT_COLUMN] > 1.0).sum()),
        content_sha256=written.provenance.outputs.table_parquet.content_sha256,
        directory=directory,
    )


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


# --------------------------------------------------------------------- main


def format_report(reports: Iterable[EndpointReport]) -> str:
    """A fixed-width summary of a run, one line per endpoint."""
    header = (
        f"{'dataset_id':32s} {'task':16s} {'metric':9s} {'source':>7s} "
        f"{'final':>7s} {'inval':>6s} {'filt':>6s} {'dup':>6s} {'tie':>4s} "
        f"{'aggr':>5s} {'train':>7s} {'valid':>6s} {'test':>6s} {'straddle':>8s} "
        f"{'content_sha256':14s}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.dataset_id:32s} {report.task_name:16s} "
            f"{report.headline_metric.value:9s} {report.source_molecules:7d} "
            f"{report.final_rows:7d} {report.dropped.get('invalid_smiles', 0):6d} "
            f"{report.dropped.get('smiles_filter', 0):6d} "
            f"{report.dropped.get(DUPLICATE_SMILES, 0):6d} "
            f"{report.dropped.get(DUPLICATE_LABEL_TIE, 0):4d} "
            f"{report.aggregated_rows:5d} "
            f"{report.per_split.get('train', 0):7d} "
            f"{report.per_split.get('valid', 0):6d} "
            f"{report.per_split.get('test', 0):6d} "
            f"{report.straddling_constitutions:8d} "
            f"{report.content_sha256[:14]:14s}"
        )
    return "\n".join(lines)


def run(config: TdcPreparerConfig, repository_root: Path) -> list[EndpointReport]:
    """Prepare every selected endpoint. Returns one report each."""
    group = load_admet_group(config.raw_root)
    metrics = tdc_default_metrics()
    preparer = preparer_record(repository_root)
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
        reports = run(config, REPOSITORY_ROOT)
    except TdcPreparationError as error:
        logger.error("%s", error)
        return 1
    print(format_report(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
