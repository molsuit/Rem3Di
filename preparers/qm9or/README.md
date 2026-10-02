# QM9-OR — molecule table and author-created train/validation/test splits

The QM9-OR chirality experiments of the Rem3Di paper (R/S configuration and optical-rotation-sign
prediction) train on one table: 117,625 molecules with labels and seven split columns. Nothing of
that table is committed. `remedi_prepare_qm9or` rebuilds it from the public Zenodo download and
checks it byte for byte against sha256 pins, so this directory holds only code and hashes.

## Source dataset

- **QM9-OR**, Zhou et al. (2025), "QM9-OR: DFT Optimized Geometries and Optical Rotations for
  Selected QM9 Molecules", Zenodo record [13380412](https://zenodo.org/records/13380412). The
  record has one file, `qm9-or.npy` (237,092,369 bytes, md5 `d7a24d28e5f611acefb9a07ff9831efd`,
  sha256 `cbe1c0def10a57e3d29c33e666b9fec30c04cf8cf3ddd8f59f642e4a1772e9ba`): 121,416 dict
  entries with a QM9 `index`, an `inchi`, a zero-padded `(27, 8)` `xyz` array (x, y, z and a
  one-hot H/C/N/O/F atom type), the per-centre CIP `chiral_centers` and the optical `rotation`
  at 355, 589.3 and 633 nm.
- The preparer downloads it into the gitignored `benchmark_data/raw/qm9or/` when it is missing
  and refuses a file whose size, sha256 or md5 differ.

## Run

From the `Rem3Di` root:

```bash
uv run --no-sync prepare-qm9or                  # once the workspace member is installed
PYTHONPATH=preparers/qm9or uv run --no-sync python -m remedi_prepare_qm9or.prepare
uv run --no-sync pytest preparers/qm9or         # synthetic tests + the full rebuild if downloaded
```

The CLI verifies the rebuild and prints a summary; it writes nothing except the raw download.
In code, `build_dataset(QM9ORPreparerConfig())` returns a `QM9ORDataset`: the `table`
(a `pandas.DataFrame`), `frames` (one `ase.Atoms` per table row, `frames[i]` belonging to row
`i`, with `molecule_id`, `qm9_index` and `source_entry` in `atoms.info`) and a `summary`. The full
rebuild takes about 1.5 minutes.

## The table

| column | meaning |
|---|---|
| `molecule_id` | 0-based, in order of first appearance of the molecule's non-isomeric SMILES in the npy |
| `isomeric_smiles` | RDKit canonical isomeric SMILES of the InChI |
| `n_chiral` | number of entries in `chiral_centers` (pseudo-asymmetric `r`/`s` included) |
| `rs` | `1` iff the centre labels are exactly `["S"]`, else `0` (see the open question below) |
| `or_sign_589` | `numpy.sign` of the 589.3 nm rotation, no threshold (−1 / 0 / +1; values are rounded to 0.01) |
| `random_s42 … random_s45` | seeded random split, value ∈ {`train`,`valid`,`test`} |
| `scaffold_s0 … scaffold_s2` | Bemis–Murcko scaffold split, value ∈ {`train`,`valid`,`test`} |

How it is built:

1. **One row per 2D molecule.** Entries are grouped by the RDKit non-isomeric canonical SMILES of
   their InChI; the **first** entry of each group wins (its SMILES, labels and geometry).
   121,416 entries become 117,625 molecules; 3,791 entries in 3,259 molecules are dropped.
2. **Random splits** (`generators/make_splits.py`): `numpy.random.default_rng(seed).permutation`
   over the molecule ids; the first `round(0.2 n)` are test, the next `round(0.1 n)` valid, the
   rest train. Counts per seed: **train 82,338 / valid 11,762 / test 23,525**.
3. **Scaffold splits** (`generators/make_scaffold_splits.py --algo shuffle`): RDKit
   `MurckoScaffoldSmiles(includeChirality=True)` of `isomeric_smiles`, 19,074 scaffold groups in
   order of first appearance, visited in `default_rng(seed).permutation` order (seeds 0–2) and
   packed whole into train (≤ 0.8 n), then valid (≤ 0.9 n), then test. Counts per seed:
   **train 94,100 / valid 11,762 / test 11,763**. **Correction:** the frozen columns use
   `algo="shuffle"`, not the generator's default `balanced` (which agrees on only 63–68 % of
   rows). The scaffold columns were earlier described as frozen partitions that might not be
   reproducible; they are reproduced exactly.

Both families are keyed by `molecule_id`, so all stereoisomers/enantiomers of a molecule land in
the same partition, and no scaffold spans two partitions.

## Pins

All three are `pandas.DataFrame.to_csv(index=False)` serialisations of the rebuilt table:

| what | line ending | sha256 |
|---|---|---|
| full table (the former committed `qm9or_splits.csv`, 9 MB) | CRLF | `b5592028060841535f8c7607d5ac00b6089b74c27987914285641130806149ac` |
| every column except `isomeric_smiles` | CRLF | `dafdd77158ce2f86d1fbb9864b9ba76234432316849024448cae3450cbcb392f` |
| first five columns (the author's lost `qm9or_molecules-use.csv`) | LF | `38386849432b81390454748dbda9ccb5ac11c260e621ba848b86a469266be2d6` |

A mismatch, or wrong molecule or fold counts, fails the build. When only the second hash
matches, the message says so: only the SMILES strings differ, i.e. the installed RDKit
canonicalises differently from RDKit 2026.03.5, which produced the pins. Ids, labels and
splits do not depend on the SMILES spelling.

## Geometry

Each row's frame is the kept entry's `xyz` with the padding rows removed (they must be all zero
and come after the real atoms). It is checked against the InChI molecule with explicit
hydrogens:

- **Heavy atoms** must agree, or the build fails. They agree for all 121,416 entries.
- **Hydrogen counts** are only counted: for 1,514 kept molecules (1,571 of all entries) the DFT
  geometry and the InChI differ by 1–5 hydrogens. These are radical or protonation layers that
  RDKit completes with implicit hydrogens, so for those rows the SMILES and the geometry are not
  the same species. The ids are in `QM9ORDataset.hydrogen_mismatch_molecule_ids`.
- **Duplicate entries disagree on the 589 nm sign:** in 1,550 molecules the entries include both
  a positive and a negative rotation (1,681 do not all share one sign). These are separate QM9
  geometries of one molecule. The table, like the paper, uses the first entry only; the ids are
  in `QM9ORDataset.opposite_sign_molecule_ids`.

Mirror images are **not** materialised yet (see below).

## Reference generators

`generators/` keeps the author's original scripts verbatim: `make_splits.py` (random) and
`make_scaffold_splits.py` (scaffold). They import a module that no longer exists
(`threedscriptors.data_handling.qm9or.transforms`, only used to fit an optical-rotation
standardiser) and read a zarr store and source CSV that are gone. The preparer vendors only
their random path and the `shuffle` scaffold path.

## Open question: what the `rs` label means (raised 2026-10-02, unresolved)

The `rs` column is not a per-molecule R/S label. It was derived (in the lost
`qm9or_molecules-use.csv`) from the per-centre CIP labels in Zhou et al.'s `qm9-or.npy` as
`rs = 1` iff the centre labels are exactly `["S"]`, else `0`. The source data is complete; the
collapse happened in this derived table. Breakdown over the 117,625 molecules:

| chiral centres | `rs = 0` | `rs = 1` |
|---|---|---|
| 0 | 33,026 | 0 |
| 1 | 11,131 | 11,392 |
| 2 or more | 62,076 | 0 |

So `rs = 0` means "single R centre", "no chiral centre" *or* "several centres"; it is a real R/S
label only for the 22,523 single-centre molecules (and even there `0` also covers pseudo-asymmetric
`r`/`s`). The paper (`paper.tex:480`) trains the R/S head on each molecule plus its mirror image
"with the labels flipped". If that flip was `1 - rs` over all rows, the mirror of an achiral
molecule (the same molecule) is labelled 1 and a multi-centre mirror gets 1, so the head can score
by recognising mirror copies rather than handedness. The code that trained and scored the R/S head
has not been found (the old `3DMolecularDescriptors` repository has only the OR-sign scripts).

Until this is resolved:

- the preparer reproduces `rs` exactly as committed, so the frozen table stays bit-identical;
- mirror rows are **not** materialised and no R/S relabelling is done;
- candidate fix, not applied: `rs` from the per-centre labels, defined only for single-centre
  R/S molecules (NaN elsewhere), flipped on their mirror rows; `or_sign_589` is defined for every
  molecule and unaffected. The paper's R/S numbers would then need re-running.

Also noted: the frozen `scaffold_s*` columns come from `generators/make_scaffold_splits.py` with
`algo="shuffle"`, not its default `balanced` (which matches only 63–68 % of rows).
