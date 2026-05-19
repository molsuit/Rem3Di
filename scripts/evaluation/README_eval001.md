# EVAL-001 — TDC ADMET + MoleculeNet benchmark panel

`run_eval_001.py` scores a descriptor (ECFP, or **your own model's descriptors**)
on the TDC ADMET-Group and MoleculeNet panels with apples-to-apples fairness:

- **Official splits.** TDC uses PyTDC's `admet_group` train/valid/test + official
  per-task metric (AUROC / AUPRC / Spearman / MAE — not a single metric for all).
  MoleculeNet uses the deterministic DeepChem scaffold split
  (`--molnet-splitter deepchem`, the Uni-Mol/MolCLR consensus).
- **Honest coverage.** The split is computed on the raw release, then mapped into
  the native zarr. A cell whose scored fraction of the official test split is
  below `coverage_thresholds` is flagged `coverage_limited` and must be excluded
  from win counts — it is reported, not silently imputed.

## Install

```bash
uv sync --extra mols --extra eval
```

`mols` provides molfeat (ECFP); `eval` provides lightgbm (eager import) + PyTDC
(TDC suite) + pandas/scipy/scikit-learn/pyyaml.

## Configure

```bash
cp configs/eval/eval_001_template.yaml configs/eval/eval_001_local.yaml
# edit the <PLACEHOLDER> paths
```

## Quick start (ECFP baseline)

```bash
python scripts/evaluation/run_eval_001.py \
  --config configs/eval/eval_001_local.yaml \
  --suite all --descriptor ECFP --heads linear lightgbm \
  --limit 1            # one task per suite, for a fast smoke
```

## Evaluate your own model (the bring-your-own-descriptor contract)

You provide one `.npz` per dataset; `run_eval_001.py` does the split, head fit,
official metric and coverage audit.

**Layout.** For row name `MyModel`:

```
<descriptor_cache_root>/<dataset>/MyModel.npz
```

`<dataset>` is the TDC benchmark name (e.g. `BBB_Martins`) or the MoleculeNet
dataset (e.g. `BACE-S`) — exactly the subdir names under your zarr roots.

**File format** (`threedscriptors.evaluation.eval001.descriptors.write_npz_cache`):

```python
from pathlib import Path
from threedscriptors.evaluation.eval001.descriptors import write_npz_cache

# X: float32 array, shape (N, D)
#   N == number of structures in that dataset's NATIVE zarr
#        (<tdc_root>/<dataset>/zarr  or  <moleculenet_output_root>/<dataset>/zarr)
#   row i == your model's descriptor for zarr structure i
#   ROW ORDER MUST MATCH THE ZARR STRUCTURE ORDER. The official split is mapped
#   raw-release SMILES -> standardised SMILES -> zarr row, then X[zarr_row] is
#   sliced. If your rows are not in zarr order, every metric is silently wrong.
write_npz_cache(Path(".../BBB_Martins/MyModel.npz"), X, {"model": "MyModel"})
```

Then:

```bash
python scripts/evaluation/run_eval_001.py \
  --config configs/eval/eval_001_local.yaml \
  --suite all --descriptor MyModel --heads linear lightgbm mlp --mlp-device cuda
```

## `--standardisation` (read this — silent-coverage footgun)

The raw-release SMILES → zarr-row lookup must use the SAME standardisation the
zarr was built with:

- `off24` — plain RDKit canonical (MACE-OFF24 family)
- `polar` — salt-strip + uncharge (MACE-POLAR family)
- `auto` (default) — infers from the dataset path substring (legacy behaviour)

`auto` only works if your zarr path contains `..._polar` / not. **If your cache
or zarr lives on a custom path, pass `--standardisation off24|polar`
explicitly** — guessing wrong silently scores ~40–60 % of charged-molecule
tasks (BACE-S, ClinTox, hERG) while still reporting a number.

## Scope on this branch (`fb638/add-eval-001-pipeline`, base `macepolar`)

- **Supported:** ECFP, and bring-your-own descriptors via the `.npz` contract
  above.
- **Not ported:** the REM3DI-checkpoint extraction convenience
  (`descriptors.compute_rem3di_from_zarr`) needs
  `threedscriptors.model.model_builder`, which does not exist on `macepolar`.
  Produce your descriptors with your own model and `write_npz_cache` instead —
  that is the intended path for evaluating your own model.

## What this branch changed in shared code (for review)

Additive only — no existing caller behaviour changed:

- `data_handling/dataset_creation/generators/utils.py`: **added** `standardize_mol`
  + `MACE_POLAR_ELEMENTS`; **extended** `filter_mol` with keyword-only
  `allowed_elements / allow_charged / allow_radicals / allow_isotopes /
  allow_multifragment`. Defaults preserve the original `filter_mol` behaviour
  bit-for-bit, so all 6 existing generator call sites are unaffected. EVAL-001
  opts into the strict OFF24/POLAR semantics explicitly.
- `evaluation/eval001/mapping.py`: 3 `PORT NOTE` adaptations to the `macepolar`
  `MoleculeDataset` API (no `load_smiles` kwarg) and the additive `filter_mol`.
- `pyproject.toml`: added the `eval` optional-dependency extra.

## Tests

```bash
pytest tests/test_eval001_mapping.py tests/test_eval001_metrics.py tests/test_eval001_mlp_device.py
```

`test_eval001_mapping.py` covers the integrity-critical pieces: the
duplicate-preserving disjoint source→zarr queue, the explicit standardisation
resolver, and the float32-safe TDC reconstruction tolerance.
