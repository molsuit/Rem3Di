"""Pydantic config for `scripts/dataset_creation/build_benchmark_dataset.py`.

One yaml drives the build of the full benchmark panel (every registered
MoleculeNet + TDC dataset). The runner dispatches on ``benchmark.source`` to
the matching generator, and each generator materializes its literature split
into the zarr ``split`` column at ingest. Eval may override the split at run
time. ``build_one`` skips datasets whose zarr already exists, so re-running
the script is idempotent and rebuilding a single benchmark just means deleting
its subdirectory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from threedscriptors.data_handling.dataset.tasks import ElementSet


class BenchmarkBuildConfig(BaseModel):
    """One yaml = the full benchmark panel built into ``output_root``."""

    model_config = ConfigDict(extra="forbid")

    # One subdirectory per benchmark dataset_id is written under output_root.
    output_root: Path
    moleculenet_raw_root: Path
    tdc_cache: Path

    # Per-build knobs shared across datasets.
    batch_size: int = 256
    max_atoms: int = 100
    n_sampled_conformers: int = 1
    max_embed_attempts: int = 10_000
    max_mmff_steps: int = 100
    # PyTDC ``get_train_valid_split`` default seed (test fold stays fixed). Eval
    # can re-derive the train/valid partition per seed without touching the zarr.
    tdc_train_valid_seed: int = 1

    # Molecule-standardization toggles forwarded to DatasetCreationConfig and
    # to the benchmark generators (standardize_mol → filter_mol). Defaults mirror
    # DatasetCreationConfig (strip + neutralize on, MACE-OFF24 element gate).
    strip_salts: bool = True
    neutralize: bool = True
    element_set: ElementSet = ElementSet.mace_off

    # Zarr chunk/shard layout — defaults match DatasetConfig.
    atom_chunk: int = 450
    molecule_chunk: int = 50
    atom_chunks_per_shard: int = 64
    molecule_chunks_per_shard: int = 256
    contains_smiles: bool = True
