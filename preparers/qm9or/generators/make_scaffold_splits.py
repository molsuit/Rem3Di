"""Generate N reproducible **scaffold** splits for QM9-OR, directly from the frozen source CSV.

Motivation: the original scaffold split is a single deterministic DeepChem partition -- one draw looks
cherry-picked. Here we produce ``--n_splits`` INDEPENDENT scaffold partitions using seeds
``0, 1, ..., n_splits-1`` (a clean, publishable seed convention), so the scaffold benchmark can be
reported as mean +/- std over the partitions.

Algorithm (seeded *balanced* scaffold split, keyed by ``molecule_id``):
  1. Read the frozen source CSV (``qm9or_molecules-use.csv``): molecule_id + isomeric SMILES.
  2. Bemis-Murcko scaffold per molecule_id (RDKit ``MurckoScaffoldSmiles``); group molecule_ids by
     scaffold. Whole scaffold groups are assigned atomically -> a scaffold NEVER spans train/valid/test
     (no scaffold leakage). Grouping by molecule_id means every stereoisomer/enantiomer of a molecule
     stays together (no chirality leakage).
  3. For each seed: ``np.random.default_rng(seed).shuffle(group_order)``, then greedily pack whole
     groups into train (<=train_frac), valid (<=+val_frac), rest -> test.

Leakage: the OR ``SignedLogStandardizer`` is fit on the TRAIN rows of EACH partition only and saved
per split (``or_transform_scaffold_s{K}.json``), so no target statistics leak from valid/test.

Outputs to ``<dataset>/splits/`` (all refuse to overwrite; the assignments CSV is also frozen 0444):
  * ``scaffold_s{K}.parquet``               -- molecule_id, split in {train,valid,test}
  * ``or_transform_scaffold_s{K}.json``     -- OR signed-log standardizer, train-only
  * ``scaffold_splits-use.csv``             -- molecule_id, split_s0, split_s1, ... (all splits), FROZEN
  * ``scaffold3_manifest.json``             -- counts + n_scaffolds + n_test_scaffolds_unseen + sha256
And a markdown dataset table to ``<repo>/remote-runs/qm9or-chiral-benchmark/analysis/scaffold3_datasets.md``.

Run:
    uv run python scripts/qm9or/make_scaffold_splits.py --dataset data/qm9or --n_splits 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from threedscriptors.data_handling.qm9or.transforms import SignedLogStandardizer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _refuse(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite existing {path} (delete by hand to regenerate)")


def murcko_scaffolds(smiles: list[str], include_chirality: bool) -> list[str]:
    """One Bemis-Murcko scaffold SMILES per input SMILES (empty-string scaffold for unparseable)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out: list[str] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(f"INVALID:{i}")  # unique -> lands as its own singleton group
            continue
        out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality))
    return out


def seeded_scaffold_split(
    scaffolds: list[str], seed: int, train_frac: float, val_frac: float, algo: str
) -> np.ndarray:
    """Return a per-index array of {'train','valid','test'}; whole scaffold groups stay together.

    Whole Bemis-Murcko scaffold groups are assigned atomically (no scaffold leakage), then greedily
    packed into train/valid/test by the cutoffs. The GROUP ORDER is what the seed varies:

    * ``algo="balanced"`` (Chemprop ``scaffold_balanced``, the default/publishable convention): big
      scaffold groups (larger than half the valid or test size) are placed FIRST so they land in
      train, leaving the rare/small scaffolds for valid+test -- the adversarial "test = unseen rare
      scaffolds" property. Big and small buckets are each seed-shuffled for reproducible variation.
    * ``algo="shuffle"``: a single uniform seeded shuffle of all groups (easier, higher-variance).

    Deterministic given (scaffolds, seed, fractions, algo).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    group_lists = list(groups.values())

    n = len(scaffolds)
    train_cut = train_frac * n
    valid_cut = (train_frac + val_frac) * n
    rng = np.random.default_rng(seed)

    if algo == "balanced":
        val_size = val_frac * n
        test_size = (1.0 - train_frac - val_frac) * n
        big = [g for g in group_lists if len(g) > val_size / 2 or len(g) > test_size / 2]
        small = [g for g in group_lists if not (len(g) > val_size / 2 or len(g) > test_size / 2)]
        rng.shuffle(big)
        rng.shuffle(small)
        ordered = big + small
    elif algo == "shuffle":
        ordered = [group_lists[i] for i in rng.permutation(len(group_lists))]
    else:
        raise ValueError(f"unknown algo {algo!r}")

    names = np.empty(n, dtype=object)
    n_train = n_val = 0
    for inds in ordered:
        if n_train + len(inds) <= train_cut:
            names[inds] = "train"
            n_train += len(inds)
        elif n_train + n_val + len(inds) <= valid_cut:
            names[inds] = "valid"
            n_val += len(inds)
        else:
            names[inds] = "test"
    return names.astype(str)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("data/qm9or"))
    ap.add_argument("--csv", type=Path, default=None, help="source CSV (default: <dataset>/qm9or_molecules-use.csv)")
    ap.add_argument("--n_splits", type=int, default=3, help="seeds are 0..n_splits-1")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--include_chirality", type=int, default=1, help="Murcko includeChirality (matches repo default)")
    ap.add_argument(
        "--algo",
        choices=("balanced", "shuffle"),
        default="balanced",
        help="balanced = Chemprop scaffold_balanced (big scaffolds->train, publishable default); "
        "shuffle = uniform seeded shuffle",
    )
    ap.add_argument(
        "--family",
        default=None,
        help="split family name propagated to every output file (default: scaffold_bal for "
        "algo=balanced, scaffold_shuf for algo=shuffle). Two variants coexist without clobbering.",
    )
    args = ap.parse_args()
    family = args.family or ("scaffold_bal" if args.algo == "balanced" else "scaffold_shuf")

    csv = args.csv or (args.dataset / "qm9or_molecules-use.csv")
    if not csv.exists():
        raise SystemExit(f"source CSV not found: {csv} (run export_molecules_csv.py first)")
    df = pd.read_csv(csv)
    mol_id = df["molecule_id"].to_numpy(dtype=np.int64)
    smiles = df["isomeric_smiles"].astype(str).tolist()
    assert len(np.unique(mol_id)) == len(mol_id), "source CSV must have one row per molecule_id"
    n_uniq = len(mol_id)
    seeds = list(range(args.n_splits))
    print(f"{n_uniq} molecule_ids from {csv}; seeds={seeds}")

    # scaffolds computed ONCE (reused across seeds)
    scaffolds = murcko_scaffolds(smiles, include_chirality=bool(args.include_chirality))
    n_scaffold_groups = len(set(scaffolds))
    print(f"{n_scaffold_groups} unique Bemis-Murcko scaffolds")

    # OR targets per zarr row (for the train-only standardizer fit)
    import zarr

    g = zarr.open_group(str(args.dataset), mode="r")
    row_mol_id = np.asarray(g["ids/molecule_id"][:], dtype=np.int64)
    row_targets = np.asarray(g["tasks/targets_system"][:], dtype=np.float64)  # [:,2:5] = or355/589/633

    out = args.dataset / "splits"
    out.mkdir(parents=True, exist_ok=True)

    assign = pd.DataFrame({"molecule_id": mol_id})
    manifest: dict = {
        "source_csv": str(csv),
        "source_csv_sha256": _sha256(csv),
        "n_molecule_ids": int(n_uniq),
        "n_scaffold_groups": int(n_scaffold_groups),
        "family": family,
        "include_chirality": bool(args.include_chirality),
        "algo": args.algo,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seeds": seeds,
        "files": {},
    }
    table_rows = []

    for seed in seeds:
        stem = f"{family}_s{seed}"
        pq = out / f"{stem}.parquet"
        tf = out / f"or_transform_{stem}.json"
        _refuse(pq)
        _refuse(tf)

        names = seeded_scaffold_split(scaffolds, seed, args.train_frac, args.val_frac, args.algo)
        assign[f"split_s{seed}"] = names

        counts = {k: int((names == k).sum()) for k in ("train", "valid", "test")}
        assert counts["train"] > 0 and counts["valid"] > 0 and counts["test"] > 0, counts

        pd.DataFrame({"molecule_id": mol_id, "split": names}).to_parquet(pq, index=False)

        # scaffold-leakage check: no scaffold spans >1 split
        sdf = pd.DataFrame({"scaffold": scaffolds, "split": names})
        spanning = (sdf.groupby("scaffold")["split"].nunique() > 1).sum()
        assert spanning == 0, f"{spanning} scaffolds span multiple splits (seed {seed})"

        # train-only OR standardizer
        train_mids = mol_id[names == "train"]
        row_mask = np.isin(row_mol_id, train_mids)
        SignedLogStandardizer.fit(row_targets[row_mask, 2:5]).save(tf)

        # scaffold stats for the table
        train_scaf = set(np.asarray(scaffolds)[names == "train"])
        test_scaf = set(np.asarray(scaffolds)[names == "test"])
        n_test_unseen = len(test_scaf - train_scaf)
        manifest["files"][stem] = {
            **counts,
            "n_test_scaffolds": len(test_scaf),
            "n_test_scaffolds_unseen_in_train": n_test_unseen,
            "sha256": _sha256(pq),
        }
        table_rows.append(
            {
                "split": stem,
                "train": counts["train"],
                "valid": counts["valid"],
                "test": counts["test"],
                "test_scaffolds": len(test_scaf),
                "test_scaffolds_unseen": n_test_unseen,
            }
        )
        print(f"{stem}: {counts}  test_scaffolds_unseen={n_test_unseen}")

    # frozen combined assignments CSV (all splits, one column each)
    assign_csv = out / f"{family}_splits-use.csv"
    _refuse(assign_csv)
    assign.to_csv(assign_csv, index=False)
    _freeze(assign_csv)
    manifest["assignments_csv"] = str(assign_csv)
    manifest["assignments_csv_sha256"] = _sha256(assign_csv)
    print(f"wrote FROZEN {assign_csv} (read-only)")

    manifest_path = out / f"{family}_manifest.json"
    _refuse(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {manifest_path}")

    # markdown dataset table
    tbl = pd.DataFrame(table_rows)
    md = [f"# QM9-OR scaffold splits — {family} (seeds 0..N-1, algo={args.algo}, 80/10/10)", ""]
    md.append(f"Source: `{csv}` ({n_uniq} molecule_ids, {n_scaffold_groups} Bemis-Murcko scaffolds)")
    md.append("")
    md.append(tbl.to_markdown(index=False))
    md_path = Path(__file__).resolve().parents[2] / f"remote-runs/qm9or-chiral-benchmark/analysis/{family}_datasets.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")
    print("\n" + tbl.to_string(index=False))

    # ---- self-check: assignments CSV matches parquets, no molecule_id in >1 split within a seed
    for seed in seeds:
        pq = pd.read_parquet(out / f"{family}_s{seed}.parquet")
        merged = assign[["molecule_id", f"split_s{seed}"]].merge(pq, on="molecule_id")
        assert (merged[f"split_s{seed}"] == merged["split"]).all(), f"assignments CSV mismatch seed {seed}"
        assert pq["molecule_id"].is_unique, f"molecule_id not unique in scaffold_s{seed}"
    print("self-check OK: assignments CSV matches parquets; molecule_ids unique per split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
