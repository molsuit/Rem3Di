# Chiro enantiomer-docking benchmark — ingestion + pairwise eval

Status: **implemented, tested, not yet built at full scale.** Picking-up notes
at the bottom.

## Goal

Add a stereochemical benchmark from the Chiro dataset (Adams, Pattanaik & Coley,
*Learning 3D Representations of Molecular Chirality with Invariance to Bond
Rotation Effects*, [arXiv:2110.04383](https://arxiv.org/pdf/2110.04383)) — the
**small-enantiomer docking-ranking** task — wired into the existing
`MoleculeDataset` ingest + descriptor-probe benchmark framework.

Scope decisions made this session (via AskUserQuestion):

- **Docking task only** (not the RS-classification family).
- **Build a dedicated pairwise enantiomer-ranking evaluator** (the headline
  metric is not standard regression/classification).
- **Cap conformers per stereoisomer** (default 2) to bound dataset size.

## Source data

`/p/scratch/mace/wedig1/raw_datasets/chiral_chiro_datasets/` — see its
`DATASETS_DESCRIPTION.md`. Three pandas `.pkl` DataFrames (train / validation /
test), one row per 3D conformer:

| column | meaning |
|---|---|
| `ID` | isomeric (stereo) SMILES — identifies the enantiomer |
| `SMILES_nostereo` | 2D SMILES, stereo removed — shared by the enantiomer pair |
| `rdkit_mol_cistrans_stereo` | RDKit `Mol`, one embedded 3D conformer |
| `top_score` | best (lowest) docking score, **constant across a molecule's conformers** |

Files used (the `margin3` docking splits):
`train_small_enantiomers_stable_full_screen_docking_MOL_margin3_234622_48384_24192.pkl`,
`validation_..._49878_10368_5184.pkl`, `test_..._50571_10368_5184.pkl`.

Key facts: each constitution (`SMILES_nostereo`) has exactly **2 enantiomers**;
`margin3` means the pair differs by ≥0.3 kcal/mol so every pair has a clear
winner. ~5 conformers/enantiomer. The task is **relative**: predict which
enantiomer of a pair docks better.

### Two source quirks handled in ingest

1. **No explicit hydrogens.** Stored Mols are heavy-atom only → the generator
   re-adds H with `Chem.AddHs(mol, addCoords=True)` so MACE-OFF sees complete
   molecules. Geometry is supplied (no conformer generation; QM9-style ingest).
2. **Chiral tags present on the Mols.** Collapsing to `ase.Atoms`
   (symbols+positions) drops them automatically — the model sees **coordinates
   only**, which is the honest chirality-from-geometry benchmark. The isomeric
   SMILES is used solely to assign the integer `stereoisomer_id`, never fed to
   the model.

## How enantiomer pairing falls out for free

The build orchestrator, when `contains_smiles=True`, derives the structure ids
from the `SmilesData` the generator emits:

- `molecule_id` ← canonical **non-isomeric** SMILES → **shared by the enantiomer pair** (the constitution).
- `stereoisomer_id` (`dataset.isomer_ids`) ← canonical **isomeric** SMILES → shared by all conformers of one enantiomer.

So conformers group under a stereoisomer, and the two stereoisomers group under
a constitution — exactly the structure the pairwise evaluator needs. No manual
id bookkeeping.

## The pairwise ranking metric

`threedscriptors/evaluation/benchmark/pairwise.py :: pair_ranking_accuracy`

1. Probe regresses per-conformer `top_score`.
2. Pool conformer predictions per `stereoisomer_id` (mean).
3. Group stereoisomers by `molecule_id`; for each constitution with exactly 2
   enantiomers, the predicted winner is the lower predicted score.
4. Accuracy = fraction of pairs whose predicted winner matches truth.
   **Predicted ties → 0.5** (chance). Constitutions without exactly 2
   enantiomers are skipped. Returns `(accuracy, n_pairs)`.

Why ties matter: a chirality-*invariant* descriptor gives identical features to
an enantiomer pair (same graph; mirror-image geometries share an interatomic
distance matrix) → identical predictions → ties → ~0.5. The metric reads ~0.5
for a chirality-blind representation and only climbs above chance when the
representation actually resolves the stereocentre. **That is the diagnostic the
benchmark exists to expose** — expect MACE-OFF invariant descriptors near 0.5;
ECFP from isomeric SMILES (which encodes `@`/`@@`) can do better.

## Files

New:
- `threedscriptors/data_handling/dataset_creation/generators/chiro_docking_generator.py`
  — `ChiroDockingGenerator` + `chiro_docking_task_set()` (one regression column
  `docking_top_score`).
- `threedscriptors/evaluation/benchmark/pairwise.py` — `pair_ranking_accuracy`.
- `scripts/dataset_creation/build_chiro_docking.py` — build script.
- `tests/test_chiro_docking_generator.py`, `tests/test_pairwise_ranking.py`.

Edited:
- `threedscriptors/data_handling/benchmarks.py` — new `ChiroDockingBenchmark`
  (`dataset_id="chiral_docking"`, `source="local_chiro"`), `EvalMetric.pair_ranking_accuracy`,
  `SplitVariant.predefined`; added to the `Benchmark` union / `_BY_ID` /
  `BenchmarkManifest.source` literal. Type aliases collapsed to the `Benchmark`
  union.
- `threedscriptors/evaluation/benchmark/runner.py` — `evaluate_pairwise_cell`,
  `_structure_group_ids`, and a branch in `evaluate_zarr` keyed on
  `manifest.metric == EvalMetric.pair_ranking_accuracy`; `BenchmarkResultRow.source`
  literal gained `local_chiro`.
- `threedscriptors/evaluation/framework/tasks/benchmark.py` — same pairwise
  branch in the framework eval path (`BenchmarkPanelConfig._evaluate_dataset`).

The runner branches on the **metric** (self-describing from the zarr manifest),
so both eval entry points pick up the pairwise path with no per-dataset config.

## Build it

```bash
uv run python scripts/dataset_creation/build_chiro_docking.py
# options: --raw-root, --output, --element-set {mace_off,mace_polar}, --max-conformers N
```

Defaults: raw root `/p/scratch/mace/wedig1/raw_datasets/chiral_chiro_datasets`,
output `/p/scratch/mace/wedig1/datasets/chiral_docking`, element gate `mace_off`
(H,C,N,O,F,P,S,Cl,Br,I), cap 2 conformers/stereoisomer, `min_interatomic_distance=0.5`
(guards MACE NaN). Idempotent: skips if the output dir already exists (delete to
rebuild). Writes a `benchmark_manifest.yaml` so the eval auto-discovers it.

**Not run yet** — the full build reads ~771k rows via `df.iterrows()` and is
slow; run it on a compute node / via sbatch.

## Eval it

Point an `EvalConfig` / `BenchmarkPanelConfig` `eval_root` at the output's parent
dir. Discovery reads the manifest, sees `pair-ranking-accuracy`, and dispatches
to `evaluate_pairwise_cell` for every (descriptor × learner). The result row's
`n_test` carries the **number of enantiomer pairs** scored (not conformers).

## Verification done this session

- `ruff` + `ty` clean on all new/edited files.
- New tests pass (10): generator (AddHs, conformer cap, split codes, enantiomer
  grouping round-trip) + pairwise metric (perfect/reversed/tie/unpaired/NaN) +
  an end-to-end `evaluate_pairwise_cell` with a real Ridge probe.
- Full benchmark/eval/framework/metric suites green (346 passed overall).
- Real-data smoke build: 80 rows of the test pkl → 34 structures (cap 2), H
  re-added, `mace_off` gate passes real Br-containing molecules, 8 enantiomer
  pairs grouped correctly.
- Pre-existing/unrelated failures in the repo (NOT from this work): 10 stale
  collection errors (missing `data_handling.pipelines`, `mol_id`, `dataset_io`,
  `PseudoscalarGenerator`, etc.) and 9 `test_irrep_slicing.py` failures (a
  pydantic `EmbeddingPreprocessConfig` schema mismatch from the in-progress
  `architecture_config.py` edits already in `git status`).

## Open items / next steps

- [ ] Run the full build (compute node / sbatch); sanity-check the build summary
      log (`generator kept/raw`, `FilterAtomsStage dropped`, n structures).
- [ ] Confirm `mace_off` doesn't drop chemistry we want — if any constitution
      hits a non-`mace_off` element, **both** enantiomers drop together (the gate
      is on the constitution), so pairing stays consistent; just check the
      dropped count is small. Switch to `--element-set mace_polar` if needed.
- [ ] Decide the final conformer cap (default 2). Higher = more 3D signal,
      bigger zarr.
- [ ] Add `chiral_docking` to whatever eval-panel config we run for the model +
      an ECFP(isomeric) baseline, and read off the ranking accuracy. Sanity
      expectation: invariant MACE ≈ 0.5, chirality-aware baseline > 0.5.
- [ ] (Deferred) The RS-classification family in the same raw dir is a clean
      binary 3D benchmark if we want it later — same generator pattern, standard
      classification path, no pairwise eval needed.
- [ ] (Possible) The `df.iterrows()` read is slow on the big files; vectorize or
      chunk if build wall-time is annoying.
