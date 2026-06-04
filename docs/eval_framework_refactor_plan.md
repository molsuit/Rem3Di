# Refactor: a unified, fault-tolerant eval framework with shared resources + plotter registry

## Progress

**Slice 1 — foundation + benchmark migration (DONE).**
- `threedscriptors/evaluation/results.py`: added `TableResult` / `ArrayResult`
  (with `load()` round-trip) and made `serialize_to` return manifest entries.
- New `threedscriptors/evaluation/framework/`: `resources.py` (`ResourceCache`,
  `EmbeddingSpec`/`IndexSpec`/`FingerprintSpec`), `context.py` (`EvalContext`),
  `task.py` (`EvalTask` protocol + `TaskStatus`), `plotting.py` (decoupled
  registry), `config.py` (`EvalManifest` + `TaskConfig` union), `runner.py`
  (fault-tolerant `run_manifest` → incremental artifacts + `manifest.yaml` +
  `status.yaml` + git SHA), `tasks/benchmark.py` (`BenchmarkPanelConfig` reusing
  `benchmark/runner.py` cell helpers via shared `EmbeddingSpec`).
- Deleted the broken `evaluation/tasks.py`.
- Tests: `tests/test_eval_framework.py` (fault tolerance, keep-going vs
  fail-fast, shared-resource-built-once, artifact round-trip, plotter registry).
  48 relevant tests pass; ruff + ty clean on all new/edited files.

**Slice 2 — retrieval task + unified entrypoint (DONE).**
- `framework/tasks/retrieval.py` (`RetrievalConfig`): builds the `VectorStore`
  from a shared `EmbeddingSpec` + `IndexSpec` (the two sub-tasks reuse one
  embedding + one index) instead of `build_vector_store`; reuses
  `run_tanimoto_similarity` / `run_nearest_molecule`. Added to the `TaskConfig`
  union.
- `scripts/evaluation/run_eval.py`: single entrypoint (`--config manifest.yaml`)
  replacing `run_benchmark_eval.py` + `run_retrieval_eval.py` for the unified
  path; exits non-zero if any task failed.
- `generate_pcqm_novicreg_eval_configs.py` now emits one `manifest.yaml` per
  model (model `remedi_<run>` so benchmark embeddings reuse existing caches;
  retrieval re-embeds PCQM100k once under the new key).
- `submit_eval_node.sbatch` runs one manifest per GPU via `run_eval.py`;
  `submit_eval_jobs.py` requires `manifest.yaml`.
- Test: `test_retrieval_task_tanimoto_on_cpu` (ECFP model, CPU). 38 relevant
  tests pass; ruff + ty clean.

**Slice 3 — descriptor-analysis task + decoupled plotting (DONE).**
- `framework/tasks/descriptor_analysis.py` (`DescriptorAnalysisConfig`): builds a
  `DescriptorAnalysisContext` from the shared `EmbeddingSpec` and runs the
  existing `descriptor_analysis` task family, each sub-task in its own
  try/except, artifacts re-rooted under `descriptor_analysis/`. Added to the
  `TaskConfig` union (all three active-path kinds now manifest-runnable).
- `framework/builtin_plotters.py`: first decoupled plotter
  (`@register_plotter("benchmark_results")`) — per-metric grouped bar from the
  saved `results.csv`. `scripts/evaluation/replot.py` re-renders a finished run's
  figures from its data artifacts (no eval re-run).
- Tests: descriptor-analysis capacity diagnostic on CPU; benchmark plotter from a
  results.csv. 47 relevant tests pass; ruff + ty clean.

**Remaining / deferred (optional).**
- Full decoupling of the 11 existing `descriptor_analysis` task classes into
  data-artifact + registered-plotter pairs (they currently still emit
  `FigureResult` inline; the framework re-roots them fine). Large rewrite, low
  marginal value — deferred.
- Add a normalized-vs-baseline bar + cross-model report plotters (the stale
  `plot_property_prediction.py` was removed; `builtin_plotters.py` currently has
  the per-metric per-run bar only).
- DONE: retired the legacy `run_benchmark_eval.py` / `run_retrieval_eval.py`
  scripts, the old benchmark/retrieval sbatch (`run_benchmark_eval.sbatch`,
  `run_retrieval_eval_5k.sbatch`, `run_retrieval_ablation_sweep.sbatch`), and the
  stale `configs/eval/{benchmark_eval_local,benchmark_eval_template,retrieval_pcqm100k}.yaml`.
  The `benchmark/runner.py` + `retrieval/runner.py` *modules* stay (the framework
  tasks and tests reuse their helpers / `RetrievalReport`).
- Pre-existing repo-wide test breakage (stale `DatasetConfig` import in several
  test modules) — separate cleanup, out of scope.

## Context

Evaluation today is split across **two unrelated frameworks** plus ~14 one-off
scripts in `scripts/evaluation/`:

- **Config-driven, functional** — `benchmark/runner.py` (`EvalConfig` + `run_eval`)
  and `retrieval/runner.py` (`RetrievalEvalConfig` + `run_retrieval_eval`).
- **Object-oriented** — `BaseEvalTask` + `EvalPipelineRunner`
  (`evaluation_pipeline.py`) + `EvalResult` (`results.py`), used by `regression/`,
  `chiral/`, `similarity_screening/`, `descriptor_analysis/`.

They share no config schema, output convention, result type, or runner. Three
concrete problems motivate this refactor:

1. **Not fault tolerant.** `benchmark/runner.py::run_eval` accumulates rows for
   all 37 benchmarks in memory and calls `write_results` *once at the end*; one
   throwing benchmark (or a wall-clock timeout) discards every successful
   result. `retrieval` is better (writes per task) but a throwing task aborts
   before `retrieval_report.yaml`.
2. **No shared-resource reuse.** The expensive artifacts — the descriptor
   **embedding matrix** over a dataset, and the **kNN index** over it — are
   recomputed per task. Tasks that operate on the same `(dataset, model)` should
   compute these once and share them.
3. **Plotting is entangled and not isolated per run.** Plots are produced inline
   (e.g. `plot_property_prediction.py` even `savefig`s to cwd); artifacts from
   different training runs aren't consistently isolated.

The good news: `descriptor_analysis/runner.py::DescriptorAnalysisRunner` is
**already** the target pattern — per-task `try/except`, serialize-as-you-go, a
shared `DescriptorAnalysisContext` (with an ad-hoc `cache: dict[str, object]`).
This refactor **generalizes that runner** into the canonical framework, formalizes
the `cache` into a typed lazy resource provider, and adds a decoupled plotter
registry. `EvalResult` (`results.py`) is reused as the artifact protocol.

**Scope (confirmed): active path only** — benchmark + retrieval +
descriptor_analysis. The OO task families (`regression/`, `chiral/`,
`similarity_screening/`) and `evaluation_pipeline.py` are left in place to
migrate opportunistically later.

**Plotting (confirmed): decoupled** — tasks emit pure-data artifacts; a registry
maps artifact type → plotter so figures regenerate offline from saved artifacts.

## Target architecture

New package `threedscriptors/evaluation/framework/`:

### 1. `framework/config.py` — one manifest
```python
class EvalManifest(BaseModel):
    model: RemediConfig                 # reuse benchmark/descriptors.py::RemediConfig
    output_root: Path                   # <eval_base>/<model_name>
    resource_cache_dir: Path | None     # shared npz cache (sibling of model dirs)
    tasks: list[TaskConfig]             # Annotated union, discriminator="kind"
    seed: int = 0
    keep_going: bool = True
```
`TaskConfig = Annotated[BenchmarkPanelConfig | RetrievalConfig | DescriptorAnalysisConfig, Field(discriminator="kind")]`.
Each task config names the dataset(s) it needs (path + id) so the runner can
key shared resources. Follows the repo's Annotated-union-with-`kind` convention
(per CLAUDE.md).

### 2. `framework/resources.py` — compute-once-share-per-task (the core new piece)
Formalizes `DescriptorAnalysisContext.cache`. A `ResourceCache` memoizes by a
stable key and persists where a disk form exists:
```python
class ResourceSpec(Protocol):
    def key(self) -> str: ...
    def disk_path(self, cache_dir: Path) -> Path | None: ...   # None => in-memory only
    def build(self, ctx) -> Any: ...

class ResourceCache:
    def get(self, spec: ResourceSpec) -> Any:    # in-memory memo, then disk, then build+persist
```
Concrete specs:
- `EmbeddingSpec(dataset_id, model)` → `(N,D)` matrix. `build` reuses
  `benchmark/descriptors.py::RemediCalculator.calculate`; disk form reuses the
  existing `{dataset_id}__{name}.npz` convention (`compute_and_cache`).
- `IndexSpec(embedding_key, index_config)` → fitted `RetrievalIndex` (reuse
  `vector_store.py::SklearnIndexConfig.build` + `.fit`); in-memory only.
- `FingerprintSpec(dataset_id, radius, n_bits)` → ECFP matrix for the Tanimoto
  baseline; npz-cached.

Result: within one run, the embedding and kNN index over a dataset are built
once and shared by every task that requests them (retrieval Tanimoto +
nearest-molecule share one embedding+index; a descriptor-analysis task on the
same dataset reuses the same npz).

### 3. `framework/context.py` — `EvalContext`
Generalizes `DescriptorAnalysisContext`: holds the manifest, the `ResourceCache`,
`output_root`, `seed`, and a `task_dir(kind) -> Path` helper. `DescriptorAnalysisContext`
is rebuilt as a thin view constructed from an `EmbeddingSpec` result.

### 4. `framework/task.py` — task protocol
```python
class EvalTask(ABC):
    kind: str
    def run(self, ctx: EvalContext) -> Iterator[EvalResult]: ...   # yield so artifacts flush incrementally
```
Tasks emit **pure-data** artifacts (no figures), via new data-artifact types in
`results.py`: `TableResult` (CSV, e.g. benchmark rows), `RecordResult`
(yaml/pydantic, e.g. retrieval metrics), `ArrayResult` (npz, e.g. projection +
colors). Each gains a `load(path)` classmethod so artifacts round-trip for
offline re-plotting.

### 5. `framework/plotting.py` — decoupled plotter registry
```python
PLOTTERS: dict[str, list[Plotter]]          # keyed by artifact kind/result_type
def register_plotter(artifact_kind: str): ...        # decorator
def render(artifacts: Iterable[EvalResult], out_dir: Path) -> list[FigureResult]: ...
```
A `Plotter` is `(artifact, out_dir) -> list[FigureResult]`. Existing plot code in
`descriptor_analysis/plotting.py` and `plot_property_prediction.py` is moved
into registered plotters. Enables `scripts/evaluation/replot.py --run-dir <model_dir>`
that loads saved artifacts and re-renders without rerunning the eval.

### 6. `framework/runner.py` — fault-tolerant runner
Generalizes `DescriptorAnalysisRunner.run`:
- Per task `try/except`; **within `BenchmarkPanelTask`, per `(dataset × learner)`
  cell** so one bad benchmark doesn't sink the panel.
- Serialize artifacts as each yields; append to `results.csv` incrementally.
- Record `TaskStatus{name, kind, ok, error, traceback, duration, artifacts}`.
- `finally:` always write `manifest.yaml` (resolved config + git SHA) and
  `status.yaml` (per-task/per-cell outcomes) at `output_root` — this is the
  "deliver most artifacts even if one breaks" guarantee.
- **Idempotent resume**: skip a task/cell whose artifact exists and status==ok
  (combined with the npz cache, resubmit after timeout finishes in minutes).
- Honors `keep_going` (default) vs fail-fast.

### 7. Per-run artifact isolation (enforced by the runner)
```
<eval_base>/
  descriptor_cache/                 # shared resources (npz), sibling of models
  <model_name>/
    manifest.yaml  status.yaml
    benchmark/results.csv
    retrieval/{tanimoto_similarity,nearest_molecule,retrieval_report}.yaml
    descriptor_analysis/<artifacts>
    plots/<rendered figures>
```

## Migration of the active path

- **`benchmark/runner.py`**: move `evaluate_zarr` logic into `BenchmarkPanelTask`;
  descriptors come from `ctx.resources.get(EmbeddingSpec(...))`; per-cell
  try/except + incremental CSV append. Keep `metrics.py`, `learners.py` as-is.
- **`retrieval/runner.py`**: `run_retrieval_eval` → `RetrievalTask`; the embedding
  + index come from `EmbeddingSpec`/`IndexSpec` (so `build_vector_store` splits
  into those two builders); tasks emit `RecordResult`s.
- **`descriptor_analysis`**: register its tasks under the new runner; build its
  `DescriptorAnalysisContext` from the shared `EmbeddingSpec` result; move inline
  plotting into registered plotters.
- **Entrypoints**: new `scripts/evaluation/run_eval.py --config manifest.yaml`
  replaces `run_benchmark_eval.py` + `run_retrieval_eval.py`; add
  `scripts/evaluation/replot.py`.
- **Generator + sbatch**: `generate_pcqm_novicreg_eval_configs.py` emits one
  `manifest.yaml` per model; `submit_eval_node.sbatch` calls the single
  entrypoint once per model per GPU (drops the `benchmark && retrieval &&` chain
  → one resumable run). The 4-per-node packing and `submit_eval_jobs.py` are
  unchanged.

## Cleanup
- Delete `evaluation/tasks.py` (raises `NotImplementedError` at import — dead/broken).
- Fix `EvalResult.serialize_to` to return the documented manifest entry (used to
  assemble `status.yaml` artifact lists; subclasses currently return `None`).
- Leave `evaluation_pipeline.py` + OO task families untouched (out of scope).

## Critical files
- Reuse/extend: `evaluation/results.py`, `evaluation/descriptor_analysis/{runner,context,plotting}.py`,
  `evaluation/retrieval/vector_store.py` (`build_vector_store`, `SklearnFlatIndex`),
  `evaluation/benchmark/{descriptors,learners,metrics}.py`.
- Refactor: `evaluation/benchmark/runner.py`, `evaluation/retrieval/runner.py`.
- New: `evaluation/framework/{config,resources,context,task,plotting,runner}.py`.
- Scripts: new `run_eval.py`, `replot.py`; edit `generate_pcqm_novicreg_eval_configs.py`,
  `submit_eval_node.sbatch`.

## Verification
- **Tests** (`tests/test_eval_framework.py`, pytest + pytest-cov per CLAUDE.md):
  1. *Fault tolerance*: synthetic tiny zarr fixture + a task/cell that raises →
     assert survivors' artifacts + `status.yaml` land, failed unit recorded with
     traceback, run does not abort.
  2. *Shared resource*: two tasks requesting the same `EmbeddingSpec` → builder
     invoked once (counting spy).
  3. *Resume*: rerun skips completed cells (builder/learner not re-invoked).
  4. *Plotter registry*: register a plotter, load an artifact from disk, render →
     figure file written.
- **Lint**: `uv run ruff check` + `uv run ty check` on new/edited files.
- **Smoke (GPU, develbooster)**: a manifest for one model over a 2-benchmark
  subset + capped retrieval (`max_structures`) → confirm `status.yaml` all-ok,
  `results.csv` present, artifacts isolated under `<model>/`, embedding npz
  written once, rerun fast (resume), `replot.py` regenerates figures from
  artifacts. Then force one benchmark to fail (mixed task types) → confirm the
  panel still writes the other rows and `status.yaml` marks the one failure.
