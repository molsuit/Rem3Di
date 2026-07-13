"""Dump every curated polaris-hub dataset to standardized parquets.

``polaris-lib`` pins ``zarr<3`` whereas this project requires ``zarr>=3.2`` for
the MoleculeDataset store, so the two cannot share a Python env. This script
sidesteps the conflict by declaring its deps inline (PEP 723) so ``uv run``
materializes a polaris-only ephemeral env per invocation -- the parent project
env is never touched.

Usage::

    uv run scripts/dataset_download/dump_polaris.py --out-root /path/to/polaris_raw

    # Re-download a dataset even if its parquet already exists.
    uv run scripts/dataset_download/dump_polaris.py \\
        --out-root /path/to/polaris_raw --force

For each entry in ``_DATASETS`` a parquet ``<dataset_id>.parquet`` is written
under ``--out-root`` with a fixed schema consumed by ``PolarisOfflineGenerator``:

* ``smiles`` -- str
* ``split`` -- uint8 Split code (0=train, 1=valid, 2=test, 255=unassigned)
* one column per source task (column names preserved verbatim so the package
  registry can reference them as ``BenchmarkTask.name``)

Existing parquets are skipped (idempotent re-runs); pass ``--force`` to
re-dump. Polaris-hub benchmark endpoints are intentionally not used here:
they are scored subsets of the underlying datasets, so we fetch the dataset
directly and let the package registry pick the scored task subset.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polaris-lib>=0.13",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Split code matches remedi.data_handling.dataset.tasks.Split. Hardcoded
# (not imported) so this script stays free of the remedi package.
_SPLIT_TRAIN, _SPLIT_VALID, _SPLIT_TEST, _SPLIT_UNASSIGNED = 0, 1, 2, 255

# Polaris ``Set`` labels seen across ASAP / Biogen / polaris-org datasets.
_SET_TO_CODE = {
    "train": _SPLIT_TRAIN,
    "valid": _SPLIT_VALID,
    "validation": _SPLIT_VALID,
    "val": _SPLIT_VALID,
    "test": _SPLIT_TEST,
}

# Columns that are never task targets (metadata / id / split annotation).
_NON_TASK_COLUMNS = {
    "Molecule Name",
    "Set",
    "CXSMILES",
    "smiles",
    "SMILES",
    "Smiles",
    "MOL_smiles",
    "UNIQUE_ID",
    "MOL_smiles_index",
    # Polaris bookkeeping that occasionally surfaces as a column:
    "MOL_molhash_id",
}

# Curated polaris datasets. (dataset_id, polaris-hub slug, source SMILES column.)
# Matches ``POLARIS_BENCHMARKS`` in ``remedi/data_handling/benchmarks.py``;
# add new entries to both places.
_DATASETS: tuple[tuple[str, str, str], ...] = (
    (
        "polaris_antiviral_admet",
        "asap-discovery/antiviral-admet-2025-unblinded",
        "CXSMILES",
    ),
    (
        "polaris_antiviral_potency",
        "asap-discovery/antiviral-potency-2025-unblinded",
        "CXSMILES",
    ),
    ("polaris_adme_fang", "biogen/adme-fang-v1", "MOL_smiles"),
    ("polaris_pkis2_subset", "polaris/drewry2017-pkis2-subset-v2", "MOL_smiles"),
)


def _split_codes_from_set(set_col: pd.Series) -> np.ndarray:
    codes = np.full(len(set_col), _SPLIT_UNASSIGNED, dtype=np.uint8)
    if set_col.empty:
        return codes
    normalized = set_col.astype(str).str.strip().str.lower()
    for label, code in _SET_TO_CODE.items():
        codes[(normalized == label).to_numpy()] = code
    return codes


def _load_polaris_table(slug: str) -> tuple[pd.DataFrame, list[str]]:
    """Fetch a polaris dataset and materialize it as a pandas DataFrame.

    Polaris is imported lazily so the rest of the file can be byte-compiled
    in the main project venv (which intentionally lacks ``polaris-lib``).
    """
    import polaris as po
    from polaris.dataset import DatasetV1, DatasetV2

    dataset = po.load_dataset(slug)
    columns = list(dataset.columns)
    if isinstance(dataset, DatasetV1):
        df = pd.DataFrame(dataset.table[:])
    elif isinstance(dataset, DatasetV2):
        df = pd.DataFrame({c: dataset.zarr_data[c][:] for c in columns})
    else:
        raise SystemExit(f"Unexpected polaris dataset class: {type(dataset).__name__}")
    return df, columns


def _dump(dataset_id: str, slug: str, smiles_column: str, out_path: Path) -> None:
    df, columns = _load_polaris_table(slug)

    if smiles_column not in columns:
        raise SystemExit(
            f"{dataset_id}: SMILES column {smiles_column!r} not found. "
            f"Available: {columns}"
        )

    if "Set" in columns:
        split_codes = _split_codes_from_set(df["Set"])
    else:
        logger.warning(
            "%s: no 'Set' column; emitting all rows as 'unassigned'. "
            "The build pipeline will need a fallback splitter for this dataset.",
            dataset_id,
        )
        split_codes = np.full(len(df), _SPLIT_UNASSIGNED, dtype=np.uint8)

    task_columns = [c for c in columns if c not in _NON_TASK_COLUMNS]
    if not task_columns:
        raise SystemExit(
            f"{dataset_id}: no task columns inferred from {columns!r}. "
            "Update _NON_TASK_COLUMNS in this script."
        )

    out_df = pd.DataFrame({"smiles": df[smiles_column].astype(str).to_numpy()})
    out_df["split"] = split_codes
    for c in task_columns:
        out_df[c] = pd.to_numeric(df[c], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    counts = {
        "train": int((split_codes == _SPLIT_TRAIN).sum()),
        "valid": int((split_codes == _SPLIT_VALID).sum()),
        "test": int((split_codes == _SPLIT_TEST).sum()),
        "unassigned": int((split_codes == _SPLIT_UNASSIGNED).sum()),
    }
    logger.info(
        "%s: %d rows -> %s (splits=%s, %d tasks)",
        dataset_id,
        len(out_df),
        out_path,
        counts,
        len(task_columns),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Directory under which <dataset_id>.parquet is written for each entry.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-dump even if the parquet already exists (default: skip).",
    )
    args = parser.parse_args()

    for dataset_id, slug, smiles_column in _DATASETS:
        out_path = args.out_root / f"{dataset_id}.parquet"
        if out_path.exists() and not args.force:
            logger.info(
                "%s: %s exists; skipping (pass --force to re-dump).",
                dataset_id,
                out_path,
            )
            continue
        _dump(dataset_id, slug, smiles_column, out_path)


if __name__ == "__main__":
    sys.exit(main())
