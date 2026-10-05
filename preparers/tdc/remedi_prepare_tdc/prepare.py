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

import logging
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
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
    DatasetSpec,
    EvalMetric,
    FileHash,
    LabelColumn,
    PreparerRecord,
    SourceRecord,
    merge_source_rows,
    metrics_with_headline,
    sha256_of_file,
)
from remedi.data_handling.bundle.preparation import (
    PREPARER_REPOSITORY,
    BundleReport,
    PreparationError,
    PreparerSettings,
    check_unique,
    preparer_main,
    smiles_bundle_spec,
    write_smiles_bundle,
)
from remedi.data_handling.dataset.tasks import TaskType

logger = logging.getLogger("preparers.tdc")

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

#: ``tdc.metadata.admet_metrics`` values -> the bundle's metric enum.
TDC_METRIC_TO_EVAL_METRIC: dict[str, EvalMetric] = {
    "mae": EvalMetric.mae,
    "spearman": EvalMetric.spearman,
    "roc-auc": EvalMetric.auroc,
    "pr-auc": EvalMetric.auprc,
}


class TdcPreparationError(PreparationError):
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
            check_unique(
                [getattr(endpoint, field_name) for endpoint in self.endpoints],
                f"{field_name} values",
            )
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


class TdcPreparerConfig(PreparerSettings):
    """Knobs of one preparer run."""

    #: Directory handed to ``admet_group(path=...)``; holds ``admet_group/``.
    raw_root: Path = DEFAULT_RAW_ROOT
    bundle_root: Path = DEFAULT_BUNDLE_ROOT

    def dataset_ids(self) -> list[str]:
        return [endpoint.dataset_id for endpoint in TDC_ENDPOINTS]

    def selected_endpoints(self) -> list[TdcEndpoint]:
        wanted = set(self.selected_dataset_ids())
        return [endpoint for endpoint in TDC_ENDPOINTS if endpoint.dataset_id in wanted]


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


# ------------------------------------------------------------- the endpoint


def build_spec(
    endpoint: TdcEndpoint, tdc_metric: str, split_columns: list[str]
) -> DatasetSpec:
    """``dataset.yaml`` for one endpoint."""
    headline = headline_metric(tdc_metric)
    task_type = task_type_for_metric(headline)
    return smiles_bundle_spec(
        dataset_id=endpoint.dataset_id,
        description=(
            f"TDC ADMET benchmark group endpoint {endpoint.dataset_id} "
            f"({endpoint.property_description}), from tdcommons.ai. Rows are the "
            "union of the official train_val and test folds after the Rem3Di "
            "SMILES filter, one row per stereoisomer with replicate "
            "measurements aggregated; the fixed TDC test fold plus five seeded "
            "train/valid partitions are frozen as the split columns."
        ),
        labels=[LabelColumn(name=endpoint.label_name, task_type=task_type)],
        metrics=metrics_with_headline(headline, task_type),
        split_columns=split_columns,
        source_kind=SOURCE_KIND,
    )


def prepare_endpoint(
    endpoint: TdcEndpoint,
    config: TdcPreparerConfig,
    group: Any,
    tdc_metric: str,
    preparer: PreparerRecord,
) -> BundleReport:
    """Read, clean, split, aggregate and write one endpoint.

    The split columns are PyTDC's own partitions of the source rows; each
    bundle row takes the split of the source row it was first seen at.

    Raises:
        TdcPreparationError: on a source problem this preparer detects.
        LabelAggregationError: if a classification label is neither 0 nor 1.
        BundleValidationError: if the table violates a format invariant.
    """
    tdc_directory_name = _resolve_tdc_directory_name(group, endpoint.dataset_id)
    source = read_source_table(config.raw_root, tdc_directory_name)
    spec = build_spec(endpoint, tdc_metric, config.split_columns())
    raw_labels = source.labels()
    missing = sum(1 for value in raw_labels if math.isnan(value))
    if missing:
        raise TdcPreparationError(
            f"{missing} source rows have no label; TDC always supplies one"
        )
    merged = merge_source_rows(
        source.raw_smiles(),
        {endpoint.label_name: raw_labels},
        {endpoint.label_name: spec.labels[0].task_type},
        config.smiles_filter,
    )
    raw_split_values = raw_split_columns(
        group, endpoint.dataset_id, source, config.seeds
    )
    return write_smiles_bundle(
        spec=spec,
        merged=merged,
        split_values={
            column: [values[row] for row in merged.first_source_rows]
            for column, values in raw_split_values.items()
        },
        settings=config,
        preparer=preparer,
        source=SourceRecord(
            files=source_file_hashes(config.raw_root, tdc_directory_name),
            package_versions=package_versions(),
        ),
    )


def run(config: TdcPreparerConfig) -> list[BundleReport]:
    """Prepare every selected endpoint. Returns one report each."""
    group = load_admet_group(config.raw_root)
    metrics = tdc_default_metrics()
    preparer = PreparerRecord.for_script(PREPARER_REPOSITORY, Path(__file__))
    reports: list[BundleReport] = []
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


def main(argv: Sequence[str] | None = None) -> int:
    return preparer_main(
        argv,
        description=__doc__ or "",
        settings_type=TdcPreparerConfig,
        run=run,
        add_arguments=lambda parser: parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[1, 2, 3, 4, 5],
            help="the get_train_valid_split seeds to freeze",
        ),
        extra_settings=lambda arguments: {"seeds": list(arguments.seeds)},
    )


if __name__ == "__main__":
    sys.exit(main())
