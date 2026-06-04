# PCQM VICReg-disabled ablation — downstream evaluation results

## Context

This documents the downstream evaluation of the 11 VICReg-disabled PCQM
pretraining runs in `/p/scratch/mace/wedig1/training_runs/pcqm_ablation_novicreg`
(trained 15 epochs on `pcqm500k_std`, descriptor shaped purely by the
atom-denoising objective). For each run we evaluated:

- **Property prediction** — REM3DI descriptor × {Ridge/LogReg linear probe, small
  MLP} over 37 prepared benchmark zarrs (MoleculeNet + TDC + Polaris), one
  independent fit per target column (multi-target sets like `polaris_adme_fang`
  scored per column).
- **Retrieval** — Tanimoto-similarity + nearest-molecule over PCQM100k
  (`pcqm100k_std`, 110k structures) embedded by each encoder.

Results live under `/p/scratch/mace/wedig1/evaluation_results/pcqm_ablation_novicreg/<model>/`
(`benchmark/results.csv`, `retrieval/*.yaml`); shared embeddings are cached in
`descriptor_cache/`. All 11 models completed with **zero benchmark failures**
(1122 result rows, 0 NaN/inf).

> Note on a prior false start: the first submission crashed on `polaris_adme_fang`
> (multi-target regression) and — because the old runner wrote results only at
> the end — lost everything. The runner is now fault-tolerant (per-dataset
> try/except, incremental `results.csv` writes, `failures.yaml`) and supports
> per-column multi-target regression. See `threedscriptors/evaluation/benchmark/runner.py`.

## 1. Property-prediction benchmark — mean rank (1 = best of 11)

Each (dataset × target × learner) cell is ranked across the 11 models
(metric-direction aware); the table averages those ranks.

| Rank | Model | Mean rank | | Rank | Model | Mean rank |
|---|---|---|---|---|---|---|
| 1 | **pcqm_agg_mean** | 3.43 | | 7 | pcqm_dropout_0.1 | 6.73 |
| 2 | pcqm_baseline | 4.03 | | 8 | pcqm_agg_pma | 7.39 |
| 3 | pcqm_depth_2 | 4.11 | | 9 | pcqm_dim_32 | 8.04 |
| 4 | pcqm_dim_128 | 4.56 | | 10 | pcqm_lr_1e-3 | 8.32 |
| 5 | pcqm_dropout_0.5 | 4.79 | | 11 | pcqm_depth_6 | 8.44 |
| 6 | pcqm_lr_1e-4 | 6.16 | | | | |

Per-cell "wins" (best of 11): agg_mean 33, dim_128 16, baseline 16, depth_2 14.
Per-cell "worst": lr_1e-3 29, depth_6 21, dim_32 17.

**Linear vs MLP:** the small MLP wins 64% of paired cells (30/47) and slightly
improves the top models (agg_mean 3.69→3.18, baseline 4.35→3.71 mean rank).
Helpful on average, not decisive; it cannot rescue the collapsed descriptors.

## 2. Retrieval over PCQM100k (random-pair Tanimoto baseline ≈ 0.083)

| Model | Spearman(emb↔Tanimoto) | Enrichment | within-Tanimoto |
|---|---|---|---|
| **pcqm_baseline** | 0.130 | 1.25 | 0.103 |
| pcqm_depth_2 | 0.117 | 1.24 | 0.102 |
| pcqm_agg_mean | 0.113 | 1.25 | 0.103 |
| pcqm_dropout_0.5 | 0.101 | 1.24 | 0.103 |
| pcqm_lr_1e-4 | 0.088 | 1.17 | 0.097 |
| pcqm_dim_32 | 0.042 | 1.05 | 0.087 |
| pcqm_dropout_0.1 | 0.038 | 1.08 | 0.090 |
| pcqm_dim_128 | 0.030 | 1.07 | 0.089 |
| pcqm_depth_6 | 0.021 | 0.99 | 0.082 |
| pcqm_agg_pma | 0.013 | 1.00 | 0.083 |
| pcqm_lr_1e-3 | 0.006 | 1.04 | 0.086 |

`depth_6`, `agg_pma`, `lr_1e-3` sit at enrichment ≈ 1.0 / Spearman ≈ 0 — their
descriptors are **collapsed to near-random for chemical similarity**.

## 3. Headline: downstream quality *inverts* the denoising-loss ranking

The two downstream tasks agree strongly (same four configs top both, same tail
bottoms both), but the ordering is the **opposite** of the denoising validation
loss. The configs that minimized the pretext objective best are the worst
descriptors:

| Model | best denoising val | bench rank | retrieval Spearman |
|---|---|---|---|
| pcqm_depth_6 | **4.7e-7** (best) | 11 (worst) | 0.021 |
| pcqm_lr_1e-3 | 9.8e-7 | 10 | 0.006 (worst) |
| pcqm_agg_pma | 9.0e-7 | 8 | 0.013 |
| pcqm_depth_2 | **4.1e-3** (worst) | 3 | 0.117 |
| pcqm_baseline | 1.6e-3 (diverged) | 2 | **0.130** (best) |

**Denoising val loss is anti-correlated with transfer quality here** — it is not
a usable model-selection signal for descriptor quality. Whether disabling VICReg
was "good" is therefore config-dependent: the best novicreg configs (agg_mean,
baseline) are genuinely strong, but several configs collapsed without VICReg —
exactly the failure mode it guards against. (A clean VICReg-on vs -off A/B is
confounded by dataset size / batch budget / hardware; not attempted here.)

## 4. Capacity diagnostic vs downstream quality

The capacity diagnostic (`descriptor_analysis/capacity_diagnostic.py`) computes
intrinsic descriptor statistics. Recomputed here on the cached PCQM100k
embeddings (CPU, no re-encoding):

| Model | D | d_eff (raw) | d_eff (l2) | bench rank | retr Spearman |
|---|---|---|---|---|---|
| pcqm_agg_mean | 64 | 0.0015 | 11.84 | 3.43 | 0.113 |
| pcqm_baseline | 64 | 0.0397 | 10.87 | 4.03 | 0.130 |
| pcqm_depth_2 | 64 | 1.030 | 1.37 | 4.11 | 0.117 |
| pcqm_dim_128 | 128 | 0.0007 | 9.10 | 4.56 | 0.030 |
| pcqm_dropout_0.5 | 64 | 1.108 | 2.04 | 4.79 | 0.101 |
| pcqm_lr_1e-4 | 64 | 0.0038 | 11.15 | 6.16 | 0.088 |
| pcqm_dropout_0.1 | 64 | 0.0013 | 6.82 | 6.73 | 0.038 |
| pcqm_agg_pma | 1024 | 0.00002 | 6.25 | 7.39 | 0.013 |
| pcqm_dim_32 | 32 | 0.00004 | 6.02 | 8.04 | 0.042 |
| pcqm_lr_1e-3 | 64 | 0.0001 | 4.14 | 8.32 | 0.006 |
| pcqm_depth_6 | 64 | 0.0002 | 7.13 | 8.44 | 0.021 |

**Aggregate Spearman across the 11 models:**

| Capacity metric | vs bench mean rank | vs retrieval Spearman |
|---|---|---|
| **d_eff (raw)** | **−0.69** (p=0.019) | **+0.81** (p=0.003) |
| d_eff (raw, top-1% norm outliers dropped) | −0.69 | +0.81 |
| d_eff (l2-normalized, direction-only) | −0.34 (p=0.31) | +0.16 (p=0.63) |
| H_tot (total marginal entropy) | −0.54 (p=0.09) | +0.15 |

Higher raw effective dimension ⇒ lower (better) benchmark rank and better
retrieval. The signal is robust to outlier removal but **vanishes once descriptor
scale is removed** (`d_eff_l2`) — i.e. the predictive information lives in the raw
covariance scale/anisotropy, not in direction-spread alone. `dead_dims` (per-dim
marginal-entropy collapse) is 0 for every model and uninformative: collapse here
is in the *covariance* (few dominant directions), not per-dimension entropy.

## 5. Per-task: is capacity predictive, task by task?

For each of 102 task-cells (62 regression, 40 classification), the across-model
Spearman between raw `d_eff` and oriented prediction quality:

| Subgroup | median ρ | mean ρ | % tasks ρ>0 |
|---|---|---|---|
| **All (102)** | **+0.53** | +0.43 | **87%** |
| Regression (62) | +0.49 | +0.38 | 82% |
| Classification (40) | +0.60 | +0.51 | 95% |
| Linear probe (51) | +0.52 | +0.39 | 82% |
| MLP (51) | +0.55 | +0.47 | 92% |

- **38% of tasks reach p<0.05, and of those 100% are positive** — when capacity
  significantly relates to quality, it is *always* "more capacity → better."
- Direction-only `d_eff_l2` is weaker per-task (median ρ +0.23, 71% positive);
  `H_tot` is intermediate (+0.39, 86%).
- Tasks where capacity most strongly predicts quality: `hERG` (ρ=0.94),
  `Pgp_Broccatelli` (0.89), `polaris_adme_fang/LOG_MDR1-MDCK_ER` (0.83),
  `CYP2C9_Veith` (0.79) — mostly classification / permeability endpoints.
- The few tasks with negative trend (high capacity slightly *hurts*) are hard
  kinase-inhibitor / ADME regressions — `polaris_pkis2_subset` RET/SLK/EGFR,
  `polaris_adme_fang/LOG_RPPB`, `VDss_Lombardo` — but **none reach significance**
  (p ≥ 0.08).

## 6. Answers to the two questions

1. **Are the capacity diagnostic tools predictive of prediction-model quality?**
   Yes. Raw `d_eff` positively predicts per-task performance on 87% of tasks
   (95% for classification) and is unanimously positive among significant tasks.
   It is a genuine, label-free early screen — *provided* you use the raw
   (scale-aware) `d_eff`, not the l2-normalized variant or `dead_dims`.

2. **Is a high-capacity representation important for prediction quality?**
   Broadly yes and rarely harmful: higher effective dimension → better prediction
   across most tasks, and never *significantly* worse. The effect is strongest as
   a floor — collapsed representations (`d_eff` → 0: depth_6, lr_1e-3, agg_pma)
   fail across the board. Among already-healthy descriptors the gradient is
   weaker and noisier (e.g. agg_mean ranks #1 on benchmarks despite a modest raw
   `d_eff`), so capacity is better read as a necessary condition than a tight
   predictor of the top of the leaderboard.

## 7. Caveats

- n=11 models per correlation → wide CIs; individual per-task ρ are noisy. The
  robust evidence is the *distribution* (87% positive; significant ⇒ positive).
- The correlation is partly driven by the collapsed-vs-healthy gap, so it
  reflects "non-collapsed ≫ collapsed" more than fine gradations among good models.
- Capacity is measured on PCQM100k (a global model property) and correlated
  against benchmark-task performance — an intrinsic-vs-extrinsic relationship.
- Mean rank compresses heterogeneous metrics; `dim_128` / `dropout_0.1` disagree
  between benchmark and retrieval, so treat the extremes as solid and the middle
  as soft.

## 8. Reproduce

```bash
# Per-model benchmark + retrieval (GPU, 4-per-node):
uv run python scripts/evaluation/submit_eval_jobs.py configs/eval/pcqm_ablation_novicreg/*

# Capacity-vs-downstream correlation (CPU, from cached embeddings):
uv run python scripts/evaluation/correlate_capacity_downstream.py \
    --eval-root /p/scratch/mace/wedig1/evaluation_results/pcqm_ablation_novicreg
# -> writes capacity_correlation/{capacity_correlation.yaml, per_task_capacity.csv}
```
