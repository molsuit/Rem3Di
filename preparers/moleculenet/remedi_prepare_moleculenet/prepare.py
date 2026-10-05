"""MoleculeNet (DeepChem S3 release) -> one bundle per dataset.

``BENCHMARK_DATA_FORMAT.md`` §5a and §11. For each of the ten datasets listed in
``datasets.yaml`` next to this module this reads one pinned csv of the DeepChem
S3 release, checks its sha256 and row count, merges the source rows into one
row per stereoisomer (:func:`remedi.data_handling.bundle.merge_source_rows`),
freezes the deterministic DeepChem scaffold test fold plus five seeded
train/valid partitions as ``split__seed{1..5}``, and writes
``benchmark_data/bundles/<dataset_id>/``.

Run from the repository root::

    uv run --no-sync prepare-moleculenet [--only esol ...] [--download]

The raw files must already be in ``benchmark_data/raw/moleculenet/``; this
script never touches the network unless ``--download`` is given, and then only
fetches the files that are missing. Every file is checked against its pin
either way.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import rdBase

from remedi.data_handling.bundle import (
    DatasetSpec,
    EvalMetric,
    FileHash,
    PreparerRecord,
    SourceRecord,
    merge_source_rows,
    sha256_of_file,
)
from remedi.data_handling.bundle.preparation import (
    PREPARER_REPOSITORY,
    BundleReport,
    PreparationError,
    PreparerSettings,
    SourceLabel,
    check_unique,
    preparer_main,
    seeded_split_notice,
    smiles_bundle_spec,
    smiles_evaluation_metrics,
    write_smiles_bundle,
)
from remedi.data_handling.chemistry.splits import (
    fixed_test_seeded_split_columns,
    scaffold_test_mask,
)
from remedi.data_handling.dataset.tasks import TaskType

logger = logging.getLogger("preparers.moleculenet")

#: The ``Rem3Di`` checkout this module lives in; the default data paths hang off it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "moleculenet"
DEFAULT_BUNDLE_ROOT = REPOSITORY_ROOT / "benchmark_data" / "bundles"
DATASETS_FILE = Path(__file__).with_name("datasets.yaml")
SOURCE_KIND = "moleculenet_deepchem_s3"

#: The MoleculeNet convention: the headline metric per task type.
HEADLINE_METRIC = {
    TaskType.regression: EvalMetric.rmse,
    TaskType.classification: EvalMetric.auroc,
}


class MoleculeNetPreparationError(PreparationError):
    """Raised when a dataset cannot be turned into a valid bundle."""


# ------------------------------------------------------------- the catalog


class MoleculeNetDataset(BaseModel):
    """One MoleculeNet bundle: which file, which columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    #: A key of :attr:`MoleculeNetCatalog.source_files`.
    source_file: str = Field(min_length=1)
    #: Read by name: tox21's SMILES column is not the first one.
    smiles_column: str = Field(min_length=1)
    #: The row count the source file must have; asserted on read.
    source_rows: int = Field(gt=0)
    task_type: TaskType
    property_description: str = Field(min_length=1)
    labels: list[SourceLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def check_labels(self) -> MoleculeNetDataset:
        if self.task_type not in HEADLINE_METRIC:
            raise ValueError(
                f"{self.dataset_id}: {self.task_type.value} is not supported"
            )
        check_unique([label.name for label in self.labels], "label names")
        check_unique([label.column for label in self.labels], "label source columns")
        return self

    def metrics(self) -> list[EvalMetric]:
        """The evaluation metrics, headline first."""
        return smiles_evaluation_metrics(
            self.task_type, len(self.labels), HEADLINE_METRIC[self.task_type]
        )


class MoleculeNetCatalog(BaseModel):
    """``datasets.yaml``: the pinned source files and every dataset, in run order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ``--download`` fetches ``<base_url><file name>``.
    base_url: str = Field(min_length=1)
    #: File name -> pinned sha256.
    source_files: dict[str, str] = Field(min_length=1)
    datasets: list[MoleculeNetDataset] = Field(min_length=1)

    @model_validator(mode="after")
    def check_consistency(self) -> MoleculeNetCatalog:
        check_unique(
            [dataset.dataset_id for dataset in self.datasets], "dataset_id values"
        )
        unpinned = sorted(
            {dataset.source_file for dataset in self.datasets} - set(self.source_files)
        )
        if unpinned:
            raise ValueError(f"source files {unpinned} have no pinned sha256")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> MoleculeNetCatalog:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


#: The ten MoleculeNet bundles (§5a).
MOLECULENET_CATALOG = MoleculeNetCatalog.from_yaml(DATASETS_FILE)


class MoleculeNetPreparerConfig(PreparerSettings):
    """Knobs of one preparer run."""

    #: Directory holding the nine DeepChem S3 files under their original names.
    raw_root: Path = DEFAULT_RAW_ROOT
    bundle_root: Path = DEFAULT_BUNDLE_ROOT
    #: The datasets and pins; tests swap in their own.
    catalog: MoleculeNetCatalog = MOLECULENET_CATALOG
    #: Fetch missing source files from ``catalog.base_url`` before the pin check.
    download: bool = False
    #: The DeepChem scaffold split ratios whose test fold is kept fixed.
    scaffold_train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    scaffold_valid_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    #: The valid share of the non-test rows in each seeded partition
    #: (1/9 of the remaining 90 % is 10 % of all rows).
    valid_fraction: float = Field(default=1.0 / 9.0, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def check_scaffold_fractions(self) -> MoleculeNetPreparerConfig:
        if self.scaffold_train_fraction + self.scaffold_valid_fraction >= 1.0:
            raise ValueError(
                "the scaffold train and valid fractions leave no test fold"
            )
        return self

    def dataset_ids(self) -> list[str]:
        return [dataset.dataset_id for dataset in self.catalog.datasets]

    def selected_datasets(self) -> list[MoleculeNetDataset]:
        wanted = set(self.selected_dataset_ids())
        return [
            dataset for dataset in self.catalog.datasets if dataset.dataset_id in wanted
        ]

    def split_notice(self) -> str:
        test_fraction = (
            1.0 - self.scaffold_train_fraction - self.scaffold_valid_fraction
        )
        return seeded_split_notice(
            "the deterministic DeepChem scaffold split (Bemis-Murcko scaffolds with "
            f"chirality, {self.scaffold_train_fraction:g}/"
            f"{self.scaffold_valid_fraction:g}/{test_fraction:g}, largest scaffold "
            "groups first) over the final bundle rows in first-occurrence order",
            self.seeds,
            self.valid_fraction,
        )


# ------------------------------------------------------------ source files


def download_missing_file(base_url: str, raw_root: Path, file_name: str) -> None:
    """Fetch ``file_name`` into ``raw_root`` unless it is already there.

    The download goes to a temporary file first so that an interrupted
    transfer never leaves a truncated file under the real name.

    Raises:
        MoleculeNetPreparationError: if the transfer fails.
    """
    target = raw_root / file_name
    if target.is_file():
        return
    raw_root.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}{file_name}"
    logger.info("downloading %s", url)
    with tempfile.NamedTemporaryFile(dir=raw_root, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as error:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise MoleculeNetPreparationError(
                f"downloading {url} failed: {error}"
            ) from error
    temporary.replace(target)


def verify_source_file(raw_root: Path, file_name: str, pinned_sha256: str) -> FileHash:
    """Check one source file against its pin and return its recorded hash.

    Raises:
        MoleculeNetPreparationError: if the file is missing or its sha256 differs.
    """
    path = raw_root / file_name
    if not path.is_file():
        raise MoleculeNetPreparationError(
            f"missing raw file {path}; put the DeepChem S3 file there or run "
            "with --download to fetch the missing files"
        )
    actual = sha256_of_file(path)
    if actual != pinned_sha256:
        raise MoleculeNetPreparationError(
            f"{path} has sha256 {actual}, but datasets.yaml pins {pinned_sha256}"
        )
    return FileHash(sha256=actual)


def ensure_source_files(config: MoleculeNetPreparerConfig) -> dict[str, FileHash]:
    """Download (if asked) and verify every file the selected datasets read."""
    file_names = dict.fromkeys(
        dataset.source_file for dataset in config.selected_datasets()
    )
    hashes: dict[str, FileHash] = {}
    for file_name in file_names:
        if config.download:
            download_missing_file(config.catalog.base_url, config.raw_root, file_name)
        hashes[file_name] = verify_source_file(
            config.raw_root, file_name, config.catalog.source_files[file_name]
        )
    return hashes


@dataclass(frozen=True)
class SourceTable:
    """The raw rows of one dataset, in file order."""

    raw_smiles: list[str | None]
    #: Bundle label name -> one value per raw row (NaN where the source is blank).
    labels: dict[str, list[float]]


def read_source_table(raw_root: Path, dataset: MoleculeNetDataset) -> SourceTable:
    """Read the SMILES and label columns of one dataset by name.

    ``HIV.csv`` as served by S3 has a blank line after every row; pandas skips
    blank lines, so the row count is still the number of molecules.

    Raises:
        MoleculeNetPreparationError: on a missing column, an unexpected row
            count, or a label value that is not a number.
    """
    path = raw_root / dataset.source_file
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise MoleculeNetPreparationError(f"cannot read {path}: {error}") from error
    wanted = [dataset.smiles_column, *(label.column for label in dataset.labels)]
    missing = [column for column in wanted if column not in frame.columns]
    if missing:
        raise MoleculeNetPreparationError(
            f"{path} has no columns {missing}; it has {list(frame.columns)}"
        )
    if len(frame) != dataset.source_rows:
        raise MoleculeNetPreparationError(
            f"{path} has {len(frame)} rows, {dataset.dataset_id} expects "
            f"{dataset.source_rows}"
        )
    labels: dict[str, list[float]] = {}
    for label in dataset.labels:
        try:
            values = pd.to_numeric(frame[label.column], errors="raise")
        except (ValueError, TypeError) as error:
            raise MoleculeNetPreparationError(
                f"{path} column {label.column!r} holds a non-numeric label: {error}"
            ) from error
        labels[label.name] = [float(value) for value in values]
    raw_smiles = [
        value if isinstance(value, str) else None
        for value in frame[dataset.smiles_column].tolist()
    ]
    return SourceTable(raw_smiles=raw_smiles, labels=labels)


# -------------------------------------------------------------- the dataset


def build_spec(dataset: MoleculeNetDataset, split_columns: list[str]) -> DatasetSpec:
    """``dataset.yaml`` for one dataset."""
    return smiles_bundle_spec(
        dataset_id=dataset.dataset_id,
        description=(
            f"MoleculeNet {dataset.dataset_id} ({dataset.property_description}), "
            f"from the DeepChem S3 release file {dataset.source_file}. Rows are "
            "the source rows after the Rem3Di SMILES filter, one row per "
            "stereoisomer with replicate measurements aggregated; the "
            "deterministic DeepChem scaffold test fold plus five seeded "
            "scaffold train/valid partitions are frozen as the split columns."
        ),
        labels=[label.label_column(dataset.task_type) for label in dataset.labels],
        metrics=dataset.metrics(),
        split_columns=split_columns,
        source_kind=SOURCE_KIND,
    )


def prepare_dataset(
    dataset: MoleculeNetDataset,
    config: MoleculeNetPreparerConfig,
    source_hash: FileHash,
    preparer: PreparerRecord,
) -> BundleReport:
    """Read, merge, split and write one dataset.

    ``source_hash`` is the verified hash of the dataset's source file
    (:func:`ensure_source_files`).

    Raises:
        MoleculeNetPreparationError: on a source problem this preparer detects.
        LabelAggregationError: if a classification label is neither 0 nor 1.
        BundleValidationError: if the table violates a format invariant.
    """
    source = read_source_table(config.raw_root, dataset)
    merged = merge_source_rows(
        source.raw_smiles,
        source.labels,
        dict.fromkeys(source.labels, dataset.task_type),
        config.smiles_filter,
    )
    is_test = scaffold_test_mask(
        merged.isomeric_smiles,
        config.scaffold_train_fraction,
        config.scaffold_valid_fraction,
    )
    return write_smiles_bundle(
        spec=build_spec(dataset, config.split_columns()),
        merged=merged,
        split_values=fixed_test_seeded_split_columns(
            merged.isomeric_smiles, is_test, list(config.seeds), config.valid_fraction
        ),
        settings=config,
        preparer=preparer,
        source=SourceRecord(
            files={dataset.source_file: source_hash},
            package_versions={"rdkit": rdBase.rdkitVersion},
        ),
        notices=[config.split_notice()],
    )


def run(config: MoleculeNetPreparerConfig) -> list[BundleReport]:
    """Prepare every selected dataset. Returns one report each.

    Every source file is verified before the first bundle is written, so a bad
    pin fails the run without leaving a partial set of bundles behind.
    """
    source_hashes = ensure_source_files(config)
    preparer = PreparerRecord.for_script(PREPARER_REPOSITORY, Path(__file__))
    reports: list[BundleReport] = []
    for dataset in config.selected_datasets():
        logger.info("preparing %s", dataset.dataset_id)
        reports.append(
            prepare_dataset(
                dataset, config, source_hashes[dataset.source_file], preparer
            )
        )
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    return preparer_main(
        argv,
        description=__doc__ or "",
        settings_type=MoleculeNetPreparerConfig,
        run=run,
        add_arguments=lambda parser: parser.add_argument(
            "--download",
            action="store_true",
            help="fetch missing source files from the DeepChem S3 release first",
        ),
        extra_settings=lambda arguments: {"download": arguments.download},
    )


if __name__ == "__main__":
    sys.exit(main())
