"""Materialise, verify, and persist the QM9-OR train/valid/test splits ONCE.

Every ablation / architecture / seed / arm / stage reads the *same* saved split files (the
SPLIT-SHARING INVARIANT), so results are comparable and reproducible. Two split families are
written to ``<dataset>/splits/``:

* ``scaffold.parquet``       -- the Bemis-Murcko scaffold split already materialised in the zarr
                                by ``QM9ORGenerator`` (no scaffold leakage).
* ``random_s{seed}.parquet`` -- a **seeded** deterministic split per replica seed (test=20% ->
                                paper-comparable; valid=10% carved for model selection; train=70%).

**Splits are keyed by ``molecule_id``** (which groups stereoisomers of the same 2D molecule), so a
molecule and all its stereoisomers/enantiomers land in the SAME split -- never any chirality
leakage across train/test. Each file has columns ``molecule_id, split in {train,valid,test}`` with
one row per unique molecule_id. A ``manifest.json`` records sizes + sha256 of every file, and for
each split we fit + save the OR signed-log standardizer on that split's TRAIN rows.

Run:
    python scripts/qm9or/make_splits.py --dataset data/qm9or --seeds 42 43 44 45
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from threedscriptors.data_handling.qm9or.transforms import SignedLogStandardizer

_CODE_TO_NAME = {0: "train", 1: "valid", 2: "test", 255: "unassigned"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_path(out: Path, stem: str) -> Path:
    p = out / f"{stem}.parquet"
    return p if p.exists() else out / f"{stem}.csv"


def _save_split(
    out: Path, stem: str, molecule_id: np.ndarray, split_name: np.ndarray
) -> Path:
    import pandas as pd

    df = pd.DataFrame({"molecule_id": molecule_id, "split": split_name})
    p = out / f"{stem}.parquet"
    try:
        df.to_parquet(p, index=False)
    except Exception:  # pragma: no cover - pyarrow/fastparquet missing
        p = out / f"{stem}.csv"
        df.to_csv(p, index=False)
    return p


def _counts(split_name: np.ndarray) -> dict:
    counts = {n: int((split_name == n).sum()) for n in ("train", "valid", "test")}
    assert counts["train"] > 0 and counts["test"] > 0, counts
    return counts


def _row_mask_for(mol_id: np.ndarray, mids: np.ndarray) -> np.ndarray:
    return np.isin(mol_id, mids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("data/qm9or"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--valid_frac", type=float, default=0.1)
    args = ap.parse_args()

    import pandas as pd
    import zarr

    g = zarr.open_group(str(args.dataset), mode="r")
    mol_id = g["ids/molecule_id"][:].astype(np.int64)  # per-row (may repeat)
    scaffold_codes = g["tasks/split"][:].astype(np.int64)  # per-row
    targets = g["tasks/targets_system"][:].astype(np.float64)  # per-row

    # collapse to unique molecule_id (groups stereoisomers)
    grp = (
        pd.DataFrame({"molecule_id": mol_id, "code": scaffold_codes})
        .groupby("molecule_id")["code"]
        .agg(lambda s: int(s.mode().iloc[0]))  # dominant scaffold code per molecule
    )
    uniq_mids = grp.index.to_numpy()
    n_uniq = len(uniq_mids)
    print(f"{len(mol_id)} rows -> {n_uniq} unique molecule_ids")

    out = args.dataset / "splits"
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "n_molecules": len(mol_id),
        "n_unique_molecule_ids": n_uniq,
        "files": {},
    }

    def finalize(stem: str, mids: np.ndarray, names: np.ndarray) -> None:
        counts = _counts(names)
        p = _save_split(out, stem, mids, names)
        train_mids = mids[names == "train"]
        SignedLogStandardizer.fit(targets[_row_mask_for(mol_id, train_mids), 2:5]).save(
            out / f"or_transform_{stem.replace('random_', '')}.json"
        )
        manifest["files"][stem] = {**counts, "sha256": _sha256(p)}
        print(f"{stem}: {counts}")

    # --- scaffold split (from the zarr, grouped per molecule) ---
    scaffold_names = grp.map(_CODE_TO_NAME).to_numpy().astype(str)
    finalize("scaffold", uniq_mids, scaffold_names)

    # --- seeded random splits (per unique molecule_id) ---
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_uniq)
        n_test = round(args.test_frac * n_uniq)
        n_valid = round(args.valid_frac * n_uniq)
        names = np.empty(n_uniq, dtype=object)
        names[perm[:n_test]] = "test"
        names[perm[n_test : n_test + n_valid]] = "valid"
        names[perm[n_test + n_valid :]] = "train"
        finalize(f"random_s{seed}", uniq_mids, names.astype(str))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
