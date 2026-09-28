# Benchmark preparers

Each preparer turns one public source into prepared benchmark bundles
(`BENCHMARK_DATA_FORMAT.md`) or, for ChiralCat, into the curated dataset a
bundle preparer reads. They used to live in the separate `remedi-data` and
`chiral_cat_cleaning` repositories and are now members of this repository's uv
workspace, each with its own `pyproject.toml`.

| directory | package | entry point | writes |
|---|---|---|---|
| `tdc/` | `remedi-prepare-tdc` | `prepare-tdc` | 22 `smiles`-stage bundles in `benchmark_data/bundles/` |
| `chiralcat/` | `chiralcat-dataset` | `build-chiralcat-dataset` | the curated dataset in `benchmark_data/curated/chiralcat/` |
| `qm9or/` | — | — | nothing yet: the frozen `qm9or_splits.csv` and the author's split generators |

## Data

Nothing under `benchmark_data/` at the repository root is committed:

```
benchmark_data/
  raw/<source>/        pinned downloads and source files a preparer reads
  bundles/<dataset>/   prepared bundles
  curated/<source>/    intermediate curated datasets (ChiralCat)
```

What stays in git is what makes the benchmarks reproducible and is small: the
preparers, the frozen split files (`qm9or/qm9or_splits.csv`,
`chiralcat/paper_splits.csv`) and the sha256 pins of every input (in each
preparer's config or provenance). The data itself is to be published
separately (build-order step 11).

## Running

From the repository root, with the venv synced once by
`uv sync --all-packages --extra cuda --extra eval`:

```
uv run --no-sync prepare-tdc [--only HIA_Hou ...]
uv run --no-sync build-chiralcat-dataset
uv run --no-sync pytest preparers/tdc
uv run --no-sync pytest preparers/chiralcat
```

Each preparer's suite runs on its own: their `conftest` modules share a name,
so one pytest session cannot collect them together.
