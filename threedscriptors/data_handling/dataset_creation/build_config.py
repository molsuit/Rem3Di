"""Pydantic config for `scripts/dataset_creation/build_benchmark_dataset.py`.

One yaml drives the build of the full benchmark panel (every registered
MoleculeNet + TDC ADMET dataset). The runner dispatches on ``benchmark.source``
to the matching generator, and each generator materializes its literature split
into the zarr ``split`` column at ingest. Eval may override the split at run
time. ``build_one`` skips datasets whose zarr already exists, so re-running
the script is idempotent and rebuilding a single benchmark just means deleting
its subdirectory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from threedscriptors.configuration.dataset_config import FilterMoleculeStageConfig


class BenchmarkBuildConfig(BaseModel):
    """One yaml = the full benchmark panel built into ``output_root``."""

    model_config = ConfigDict(extra="forbid")

    # One subdirectory per benchmark dataset_id is written under output_root.
    output_root: Path
    moleculenet_raw_root: Path
    tdc_cache: Path
    # Polaris parquet dumps (produced by scripts/dataset_download/dump_polaris.py
    # in its own venv -- polaris-lib pins zarr<3 and cannot share this env).
    # ``None`` skips the polaris panel; existing moleculenet / tdc builds keep
    # working with no config change.
    polaris_raw_root: Path | None = None

    # Per-build knobs shared across datasets. Defaults are the values
    # validated by the CYP timing experiment (slurm-4686108): three CYP_Veith
    # benchmarks went from 5.2h to 10min wall-time vs. the prior 10000/500/500
    # combination, with identical (~99%) yield. Per-mol ETKDG/MMFF timing
    # distributions are persisted alongside each zarr as
    # ``conformer_timings.jsonl`` (see ``ConformerGenerationStage``).
    batch_size: int = 256
    n_sampled_conformers: int = 1
    # ETKDG retry budget per conformer. 200 succeeds on essentially everything
    # ETKDG can solve; the legacy 10_000 default routinely sat in the tens of
    # thousands on pathological CYP rows and ate >30 min of wall time per mol.
    max_embed_attempts: int = 200
    # MMFF94 BFGS step cap per conformer.
    max_mmff_steps: int = 100
    # MMFF94 non-bonded interaction cutoff in Å. RDKit's default (100.0)
    # already covers every atom pair for drug-sized molecules; the previous
    # 500.0 setting built an effectively all-pairs neighbor list and inflated
    # per-step cost on large systems for no chemical benefit.
    mmff_non_bonded_thresh: float = 100.0
    # PyTDC ``get_train_valid_split`` default seed (test fold stays fixed). Eval
    # can re-derive the train/valid partition per seed without touching the zarr.
    tdc_train_valid_seed: int = 1

    # Optional allow-list of benchmark ``dataset_id``s. ``None`` (default) keeps
    # the legacy "build every registered benchmark" behavior. Setting it lets
    # an experiment yaml target a subset (e.g. just the CYP datasets) without
    # touching the registry or pre-populating skip-marker dirs.
    only_datasets: list[str] | None = None

    # SMILES filter knobs (max_atoms, element gate, salt-strip / neutralize,
    # dedupe). One ``FilterMoleculeStage`` instance per benchmark build is
    # constructed from this; MoleculeNet feeds the same config straight into
    # its internal ``apply_smiles_filter`` call.
    filter: FilterMoleculeStageConfig = Field(default_factory=FilterMoleculeStageConfig)

    # Zarr chunk/shard layout — defaults match DatasetConfig.
    atom_chunk: int = 450
    molecule_chunk: int = 50
    atom_chunks_per_shard: int = 64
    molecule_chunks_per_shard: int = 256
    contains_smiles: bool = True
