# ChiralCat chirality dataset

Curation pipeline that turns the raw ChiralCat source pickles into a single
labelled 3D dataset fit for training, plus an explicit record of everything that
was thrown away and why.

Class labels follow the paper (Peng et al., *AI Chemistry* 2025):

| label | class   |
|-------|---------|
| 0     | achiral |
| 1     | central |
| 2     | axial   |
| 3     | helical |
| 4     | planar  |

## Build the dataset

This package is a member of the Rem3Di uv workspace. From the Rem3Di root:

```bash
uv sync --all-packages --extra cuda --extra eval    # once
uv run --no-sync build-chiralcat-dataset
```

Everything is driven by `pipeline.yaml`; `--config` points at a different one.
`--no-write` runs the pipeline and prints the summary without touching disk.
Every input is pinned by its sha256 in `pipeline.yaml` and a mismatch aborts the
build.

Tests (the integration tests run the real pipeline over the real pickles once
per session, about a minute and a half, and are skipped when the pickles are
absent):

```bash
uv run --no-sync pytest preparers/chiralcat
```

## Output

Two datasets, in `benchmark_data/curated/chiralcat/`:

- **`dataset.csv` / `dataset.extxyz`** — the structures usable for training.
- **`rejected.csv` / `rejected.extxyz`** — everything that did not make it.
- **`run_report.json`** — per-stage counts, the corrections applied, the config.

Both CSVs carry their provenance as columns, so there are no separate audit
logs to keep in sync.

`dataset.csv`

| column | meaning |
|--------|---------|
| `index` | stable id, shared with the extxyz `index=` field |
| `smiles` | canonical isomeric SMILES of the emitted molecule |
| `label`, `class_name` | chirality class |
| `n_atoms` | atom count including explicit hydrogens |
| `source_file` | which source pickle the molecule came from |
| `geometry_quality` | `ok`, `repaired`, or `rebuilt_architector` |
| `correction` | the manual label fix applied, if any |
| `repair_strategy` | `reembed`, `strip_salt`, `strip+reembed`, `architector` |
| `note` | free-text detail |

`rejected.csv` adds `stage`, `reason`, `disposition` and `detail`.
`disposition` separates the two very different reasons a molecule is absent:

- **`broken`** — erroneous and not fixable: no parseable SMILES, no conformer,
  a dummy atom, an atom pinned at the origin, or a geometry still clashing after
  every repair strategy was tried.
- **`filtered`** — deliberately excluded: a duplicate, a `drop_types` class, or a
  `delete` entry in `corrections.yaml`.

Most rejections happen precisely because a molecule had no usable geometry, so
`rejected.extxyz` holds only the subset that still has coordinates (the
`has_geometry` column says which). The CSV is the complete record.

Read the geometries with
`ase.io.read("benchmark_data/curated/chiralcat/dataset.extxyz", index=":")`.

## Current build

**16,977 unique structures with explicit hydrogens**, deduplicated by canonical
isomeric SMILES with no label conflicts across sources:

| class | count |
|---------|-------:|
| achiral | 10,000 |
| central | 6,361 |
| axial | 522 |
| helical | 37 |
| planar | 57 |

By geometry: 16,900 untouched, 49 repaired, 28 rebuilt with Architector. Every
shipped structure is clash-free and every hydrogen sits within a bond length of
a heavy atom.

**851 rejected**, of which only 47 are broken:

| stage | reason | count | disposition |
|-------|--------|------:|-------------|
| extract | `duplicate` | 730 | filtered |
| extract | `duplicate_after_standardize` | 36 | filtered |
| repair | `flagged` | 32 | **broken** |
| extract | `dropped_type` (`unknown`) | 18 | filtered |
| repair | `dropped_duplicate` | 14 | filtered |
| extract | `bad_geometry` | 13 | **broken** |
| extract | `correction_delete` | 6 | filtered |
| extract | `dummy_atom` | 2 | **broken** |

Label corrections: 40 applied, 0 unmatched (6 relabel, 6 delete, 28 keep). The
stereo audit reports 0 uncovered mislabels.

## Frozen organometallic rebuilds

The rebuild stage does not run Architector. Architector imports xtb-python and
openbabel at module level, neither of which has wheels for Python 3.12, and its
build is not deterministic: 28 to 30 of its 30 targets succeeded from run to
run, because one borderline complex sits right on the hapticity gate. The 28
geometries of one accepted run are frozen in
`benchmark_data/raw/chiralcat/organometallic_rebuilds.extxyz`, one frame per
complex keyed by its fragmented source SMILES (`source_smiles`), and the stage
looks each target up there and re-validates it on composition, hapticity and
clashes.

The frozen file was cut on 2026-09-28 from the output of
`steffen-wedig/chiral_cat_cleaning@9b4032c`, the last commit that ran
Architector: the build was re-run up to the repair stage, each of the 30 targets
was joined to its rebuilt frame by index, and its ligand SMILES was checked
against the frame. With it, the build reproduces that output byte for byte
except for the SMILES of the 28 rebuilt rows (below) and the rejection note of
the two targets without a geometry (`rebuild no_frozen_geometry`).

A rebuilt complex's `smiles` lists its components, ligands then carbonyls then
the metal (`Cc1ccccc1CO.[C-]#[O+].[C-]#[O+].[C-]#[O+].[Cr]`). It is an identity
key, not a bonded description: SMILES has no haptic bond, and cannot express
planar chirality at all. Before 2026-09-28 it listed the ligands only, so the
(arene)Cr(CO)3 and (arene)Mn(CO)3 complex of the same arene shared a SMILES
(seven such pairs). Which planar enantiomer each frozen geometry is was chosen
by Architector, not by the source.

## Pipeline stages

`build_dataset()` chains four stages in memory — nothing intermediate is
written to disk.

1. **extract** (`chiralcat_dataset/extract.py`)
   Reads the pickles listed in `pipeline.yaml`, takes one representative
   conformer per molecule, standardizes it (`SaltRemover` +
   `rdMolStandardize.Uncharger`, both verified stereo-safe), adds explicit 3D
   hydrogens, and deduplicates on canonical SMILES. Label corrections are
   applied here, before dedup picks a winner.

2. **validate** (`chiralcat_dataset/validation.py`)
   Applies `corrections.yaml` and re-runs the RDKit stereocentre audit over the
   `central` class using `AssignStereochemistryFrom3D` on the real conformer. A
   central molecule with no 3D stereocentre that the corrections file does not
   cover is a suspected mislabel; `on_uncovered_mislabel` decides whether that
   warns, aborts the build, or is ignored. This is what keeps `corrections.yaml`
   from going stale as the sources change.

   `corrections.yaml` records three kinds of decision, all keyed by canonical
   SMILES and all discriminated on `kind`:

   | kind | effect |
   |------|--------|
   | `relabel` | the molecule is in the wrong class; move it |
   | `delete` | the molecule is chiral but the right sub-class is not clear; drop it |
   | `keep` | the audit flags it but it was reviewed and is correct as labelled |

   The `keep` kind exists because the 3D audit cannot perceive every kind of
   chirality. A P(III) phosphine (the lone pair is the fourth substituent), a
   selenoxide, and a haptic organometallic are all genuinely chiral yet carry no
   tetrahedral tag RDKit will read off the conformer. Twenty-eight such
   molecules are recorded as `keep`, so the audit stays quiet about them and a
   new warning means a genuinely new finding rather than known noise.

3. **repair** (`chiralcat_dataset/repair.py`)
   Finds atom pairs closer than an element-aware bond-length floor and tries to
   fix them, behind gates that refuse any repair which would change the
   chirality:
   - *re-embed* (ETKDGv3 + MMFF94) — only where the SMILES fully encodes the
     chirality, so only for `central` and `achiral`;
   - *strip counter-ion* — keep the largest fragment of the original conformer;
     no heavy atom moves, so the chirality geometry survives by construction.

4. **rebuild** (`chiralcat_dataset/organometallic.py`)
   The planar-chiral organometallics are stored as disconnected fragments drawn
   without the metal bonds, so their conformers have the metal sitting on the
   ring carbons and cannot be re-embedded. These are translated back into a
   metal core plus haptic ligands. Their geometries were assembled once with
   Architector and are read back from the frozen file (see above), then
   validated on composition, hapticity and clashes.

## Source data (`benchmark_data/raw/chiralcat/`)

The pickles named in `pipeline.yaml` and the frozen rebuilds are the pipeline's
only inputs; they are gitignored and published separately. Two extra CSVs were
recovered from the model repo's LMDB files before those were removed:

- `paper_splits.csv` (committed beside this README) — the paper's
  train/valid/test assignment, 9,394 rows of
  `subset, split, smiles, label, class_name, curated_index`. `subset` is
  `classifier` (the 1,631-molecule fine-tune set) or `pretrain`.
  `curated_index` joins to `index` in the curated `dataset.csv`, or is `-1` where this
  pipeline filtered the molecule out.
- `lmdb_divergent_descriptions.csv` (in the raw directory) — the 15 molecules whose LMDB description
  text differed from `*_description.csv`; the other ~7,000 matched verbatim.

## Notes

- Geometries are ETKDG embeddings optimised with MMFF94, as in the paper. There
  are no higher-level geometries in the source data: the `3d-pubchem.lmdb` files
  that shipped with the model repo were verified to hold coordinates *bitwise
  identical* to the pickle conformers (max deviation < 1e-6 Å across all 1,631
  fine-tune and 7,102 pretrain records). The B3LYP/6-31G(d) provenance sometimes
  attached to them belongs to the upstream 3D-MoLM PubChemQC pipeline and never
  applied to this data.
- Roughly 100 source molecules exist only as SMILES, with no 3D geometry, in the
  `*_description.csv` files of the raw directory. They are outside the scope of this 3D-only
  extraction and are not counted as rejected.
- The fragmented organometallic complexes that the rebuild stage cannot handle
  keep their `[Fe]`/CO fragments; those are not in RDKit's default salt list.

Original model code for the paper lives upstream at
[DantePeterson23/ChiralCat](https://github.com/DantePeterson23/ChiralCat); this
repository keeps only the data curation.
