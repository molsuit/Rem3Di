# Conformer Generation Timing Experiment

**Date:** 2026-05-21
**Branch:** `macepolar`
**Slurm jobs:** 4686108 (experiment) — reference baseline: 4671351 (production)

## Motivation

The CYP_Veith ADMET datasets dominated benchmark-build wall time. In the
2026-05-20 full-panel build, the three `CYP{2C9,2D6,3A4}_Veith` datasets
consumed 5.2 hours of `ConformerGenerationStage` time; the other twelve
benchmarks combined took ~22 minutes.

Per-batch timings on the CYP runs sat at 100–130s/batch versus ~1–9s/batch
on the other datasets. Spacing between consecutive `[HH:MM:SS] UFFTYPER:`
log lines spanned tens of minutes, consistent with single pathological
molecules hanging inside RDKit C++ for the duration. The log did not
disambiguate whether the cost was ETKDG embedding or MMFF94 relaxation —
`UFFTYPER:` fires at MMFF setup time but is silent during the actual work.

## Hypothesis

The legacy `BenchmarkBuildConfig.max_embed_attempts = 10_000` (per-conformer
ETKDG retry budget) lets pathological molecules grind through thousands of
distance-geometry attempts before failing. Step caps bound *work* but not
wall time, because per-attempt cost scales with molecule size. The expected
fix path: lower the cap, and instrument both phases to see which one was
actually paying for the tail.

## Method

1. Added per-phase timers (`time.perf_counter`) around `EmbedMultipleConfs`
   and `MMFFOptimizeMoleculeConfs` inside `embed_one_smiles`.
2. Persisted the records as `<zarr>/conformer_timings.jsonl` via a new
   `ConformerTimingRecord` pydantic model and a `flush_timings()` hook on
   `ConformerGenerationStage`, called from the orchestrator's `finalize()`.
3. Tightened two knobs simultaneously:
   - `max_embed_attempts`: 10_000 → 200
   - `mmff_non_bonded_thresh`: 500.0 → 100.0 (RDKit default for MMFF94)
4. Built only the three CYP_Veith datasets into a separate
   `benchmarks_timing_exp/` output root so production zarrs were untouched
   and the comparison was apples-to-apples.

## Results

### Wall time

| dataset | production (10k attempts / 500 Å cutoff) | experiment (200 / 100 Å) | speedup |
|---|---|---|---|
| CYP2C9_Veith | 6570s | 219s | 30× |
| CYP2D6_Veith | 6520s | 217s | 30× |
| CYP3A4_Veith | 5538s | 175s | 32× |
| **sum** | **5.2 h** | **10.2 min** | **~30×** |

### Embed vs MMFF per-molecule timings

Embedding dominates by ~6:1 in aggregate and ~100× in the tail:

| | embed | MMFF |
|---|---|---|
| p50 | 0.015s | 0.011s |
| p95 | ~0.10s | 0.024s |
| p99 | ~1.7s | 0.036s |
| max | 13–16s | 0.18s |
| sum across 3 CYPs | 2481s | 436s |

The `nonBondedThresh: 500 → 100` change was effectively free but barely
moved the needle. The win came from bounding `max_embed_attempts`.

### Yield

| dataset | total | ok | embed_failed | value_error |
|---|---|---|---|---|
| CYP2C9_Veith | 11877 | 11722 (98.7%) | 155 | 0 |
| CYP2D6_Veith | 12919 | 12769 (98.8%) | 150 | 0 |
| CYP3A4_Veith | 12125 | 11981 (98.8%) | 144 | 0 |

Failures are all `embed_failed` — the bound is doing what it should: ETKDG
gives up after 200 attempts instead of grinding for tens of minutes. Yield
matches the production builds (~99%); the molecules being dropped are the
same ones that ate the wall time before.

### Surviving slow molecules

Even with the bound in place, two structural classes still cost 10–16s per
embed:

- **Cinchona-alkaloid dimers / quinine-type frameworks** (90–100 atoms)
- **Macrolide-style stereo-heavy molecules** (128–136 atoms)

Both are pharmacologically interesting but a known ETKDG pain point. They
sit far enough out on the distribution that further bounding would have
diminishing returns relative to the dataset-wide noise.

## Decisions

The experiment values were promoted to defaults across the codebase:

- `DatasetCreationConfig`: `max_embed_attempts=200`, `max_MMFF_steps=100`
- `BenchmarkBuildConfig`: same, plus new `mmff_non_bonded_thresh=100.0` and
  an `only_datasets` allow-list knob for subset rebuilds.
- `mmff_non_bonded_thresh` is now a proper pydantic field instead of a
  hardcoded constant inside `embed_one_smiles`.
- Per-molecule timings persist as `conformer_timings.jsonl` next to every
  zarr so future builds can be compared without re-instrumentation.

## Followups

- **Wall-clock bound on stuck molecules.** If the 10–16s tail starts
  hurting at larger scale, the next step is a true per-mol wall-clock cap.
  `signal.SIGALRM` will not work — RDKit C++ releases the GIL, so Python's
  signal handler can't preempt mid-call. The right tool is
  `pebble.ProcessPool` with `timeout=`, which SIGTERMs the worker on
  overrun and respawns it.
- **nvMolKit.** NVIDIA Digital Bio's GPU MolKit (v0.5.0) was considered as
  a more aggressive alternative — batched GPU ETKDG + MMFF94 via
  `EmbedMolecules` / `MMFFOptimizeMoleculesConfs` / `MMFFBatchedForcefield`.
  Deferred: the tail is now small enough that the CUDA dependency,
  reproducibility drift (changed embedder ⇒ changed conformers in every
  dataset), and GPU-node scheduling overhead don't pay back. Reconsider if
  scaling to OMol25-sized inputs or if the cinchona/macrolide tail becomes
  load-bearing.

## Artifacts

- Experiment slurm log: `/path/to/datasets/logs/slurm-4686108.err`
- Experiment output (per-mol JSONL): `/path/to/datasets/benchmarks_timing_exp/CYP*_Veith/conformer_timings.jsonl`
- Production reference log: `/path/to/datasets/logs/slurm-4671351.err`

The bespoke experiment yaml and sbatch wrapper were removed after the
results were captured here. The `only_datasets` knob in
`BenchmarkBuildConfig` remains and can be used to target a subset rebuild
from any future config.
