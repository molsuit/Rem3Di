# Dataset Comparison Metrics

**Date:** 2026-05-27
**Branch:** `main`
**Code:** `threedscriptors/data_handling/dataset_comparison.py`,
`threedscriptors/configuration/dataset_comparison_config.py`,
`scripts/dataset_creation/run_dataset_comparison.py`

## Motivation

Before deciding whether the benchmark eval panel is already covered by the
pcqm4m pretraining set or whether the pretraining set needs to be augmented,
we need a quantitative read on how close the eval molecules sit to the
pretrain molecules in scaffold / Tanimoto space. `DatasetComparison` takes a
list of role-tagged (`pretrain` / `eval`) on-disk `MoleculeDataset` zarrs and
runs three families of metrics on their union.

Configuration is yaml-driven (`DatasetComparisonConfig`); per-dataset
subsampling (`max_molecules`) lets you probe the full panel cheaply before
committing to a multi-million-molecule comparison.

## Implemented metrics

### 1. BitBIRCH clustering on the fingerprint union

ECFP4 packed fingerprints (`bblean.fps_from_smiles`) for every unique SMILES
across all input datasets are stacked and clustered jointly with `bblean.BitBirch`
(threshold / branching_factor / merge_criterion configurable). Each row of the
union keeps its origin dataset index, so the same clustering supports both the
"is structure recovered" health checks and the role-aware contingency below.

**Health checks emitted to `dataset_comparison_summary.yaml`:**

- `n_clusters`, `n_singletons`
- `largest_cluster_size` and `largest_cluster_fraction` (largest / total) —
  a single cluster swallowing >50% of the union is a sign of too-loose
  threshold; >90% singletons signals the opposite.
- `cluster_size_distribution`: mean/std/min/quartiles/max via the shared
  `DistributionStats` model.
- `scaffold_purity_distribution`: per-cluster fraction of the most common
  Bemis-Murcko scaffold among cluster members. 1.0 = pure cluster,
  ~1/k = perfectly mixed. The mean of this distribution summarizes how well
  Tanimoto-based clusters track scaffold identity.

**Cross-dataset cluster composition:**

- `shared_clusters` (≥2 distinct datasets contribute members) vs
  `exclusive_clusters` (only one).
- `pretrain_eval_shared_clusters`: clusters containing both pretrain and
  eval molecules.
- `pretrain_molecules_in_eval_clusters` and the corresponding fraction of
  *all* pretrain molecules — how much of the pretraining set actually lives
  near the eval set under this BitBIRCH threshold.
- `eval_clusters_with_pretrain_fraction[name]` per eval dataset — out of
  this benchmark's non-empty clusters, what fraction also contain a pretrain
  molecule. Closer to 1.0 = eval set is well-blanketed by pretrain.

**Mutual information** (`sklearn.metrics`):

- `mutual_info`: raw `MI(cluster_id, dataset_name)` in nats.
- `normalized_mutual_info`: NMI ∈ [0, 1] (arithmetic-mean denominator). 0 =
  cluster assignment carries no info about which dataset a molecule came
  from (good news for coverage); 1 = clusters perfectly partition by
  dataset (eval datasets live in their own scaffold region).

**Plots:** `bitbirch_union_cluster_sizes.png` (size histogram, log-y),
`cluster_composition.png` (top-N stacked bar by dataset),
`cluster_scaffold_purity.png` (purity histogram),
`umap_by_origin.png` and `umap_by_cluster.png` (UMAP on union FPs colored
two ways), plus `cluster_contingency.npz` and
`dataset_comparison_umap_coords.npz` for re-analysis.

### 2. Per-eval nearest-neighbor Tanimoto distribution

For every eval dataset, every kept fingerprint is scored against the pooled
pretrain fingerprint set and the maximum Tanimoto is recorded. This is the
classic "is anything in pretrain similar to this eval molecule" question.

**Backends:**

- `cpu` (default): `bblean.similarity.jt_sim_packed` over packed uint8
  fingerprints, looping over query molecules. Throughput on the dev run was
  ~22M comparisons/sec → ~13 min for the full pcqm4m (3.5M) × eval (~5k)
  panel.
- `nvmolkit`: gated lazy import of `nvmolkit.fingerprints.MorganFingerprintGenerator`
  + `nvmolkit.similarity.crossTanimotoSimilarity`, computes the full (Q, R)
  similarity matrix on GPU then reduces max along the reference axis.
  Switched on with `nn_tanimoto.backend: nvmolkit` once the package is
  installed (it isn't yet — `pyproject.toml` will need a `nvmolkit` entry).

**Reported per eval dataset:**

- `n_queries`, full `DistributionStats` (mean/std/min/p05/p25/p50/p75/p95/max).
- `fraction_above_0p4` and `fraction_above_0p7` — common thresholds for
  "structurally similar" (≥0.4) and "near-duplicate" (≥0.7) in ECFP4 space.

**Plots / artifacts:** `nn_tanimoto_distributions.png` (overlaid histograms
per eval dataset), `nn_tanimoto_values.npz` with the raw per-molecule max
similarities for downstream analysis (CDFs, per-target slicing, etc.).

### 3. Bemis-Murcko scaffold set overlap

Two scaffold definitions are computed in parallel (RDKit
`MurckoScaffold.MurckoScaffoldSmiles` and a generic framework after
`MakeScaffoldGeneric`):

- `n_scaffolds_specific` and `n_scaffolds_generic` per dataset.
- `pairwise_jaccard_specific` and `pairwise_jaccard_generic`: every
  unordered dataset pair gets `|A ∩ B| / |A ∪ B|` on its scaffold sets.
  Key format `"<a>__<b>"` with `a < b` lexicographically.
- `eval_specific_coverage_by_pretrain` and the generic counterpart: per
  eval dataset, `|eval_scaffolds ∩ ⋃pretrain_scaffolds| / |eval_scaffolds|` —
  the fraction of eval scaffolds that show up *anywhere* in the pretraining
  set. This is the direct answer to "do I have these scaffolds in pretrain
  at all", complementing the Tanimoto distribution which answers "do I have
  something *close*."

**Plots:** `scaffold_jaccard_specific.png` and `scaffold_jaccard_generic.png`
heatmaps annotated with the pairwise Jaccards.

## How the three pieces fit together

| Question                                              | Metric                                          |
|-------------------------------------------------------|-------------------------------------------------|
| Do my eval molecules cluster *with* pretrain at all?  | `pretrain_eval_shared_clusters`, NMI            |
| How much of pretrain is "near" the eval set?          | `pretrain_in_eval_clusters_fraction`            |
| For each eval molecule, is there a close neighbour?   | NN-Tanimoto distribution, `fraction_above_0p4/0p7` |
| Are the *scaffolds* themselves present in pretrain?   | `eval_*_coverage_by_pretrain`                   |
| Is the eval set chemically distinct as a whole?       | Pairwise scaffold Jaccards, MI                  |

Coverage and NN-Tanimoto can disagree usefully: a low coverage but high mean
NN-Tanimoto means pretrain has *similar* but not *identical* scaffolds — a
sign that pretraining transfer can work but the eval set explores nearby
chemistry. Low on both means we need to augment the pretraining set with
that region.

## Dev verification

`configs/dataset_creation/dataset_comparison_dev.yaml` compares two small
benchmark zarrs (freesolv vs bace, role-tagged for the code paths to
exercise). Full pipeline wall time: ~50 s on a JUWELS Booster node, with
the BitBIRCH fit dominant (~48 s on 2138 molecules at threshold 0.65) and
the UMAP+plots accounting for most of the rest. NN-Tanimoto took 0.04 s,
scaffold overlap 1.2 s.

Numbers on freesolv (small aliphatic solvation molecules) vs bace (BACE-1
ligand panel) corroborate the expected story: 0 shared clusters, NMI 0.18,
NN-Tanimoto mean 0.20 (max 0.57), pairwise specific scaffold Jaccard 0.006,
eval-specific scaffold coverage by "pretrain" 0.6 %. Translation: the two
benchmark sets occupy almost disjoint chemical neighbourhoods, exactly what
you'd expect for benchmarks chosen to cover different physical
properties.

## Open work / known limits

- `nvmolkit` is not yet a declared dependency. CPU is comfortably inside
  the budget for the full pcqm4m × benchmark-panel run (~13 min) so this
  was intentionally deferred. If we end up iterating on threshold sweeps,
  flip the backend.
- BitBIRCH on the full pcqm4m (3.5M FPs) has not been timed end-to-end on
  Booster yet — the dev panel sized things to fit a login-node smoke test.
  A first full job should set `bitbirch.max_molecules` conservatively (e.g.
  500k pretrain) and confirm cluster shape before scaling up.
- The "origin label" for MI is the per-dataset name, not the binary
  pretrain/eval role — when the dataset count grows the contingency table
  also grows, and the marginal cluster entropy will push NMI down for any
  fixed cluster count. Interpret NMI changes within a fixed set of inputs.
