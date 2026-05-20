from pathlib import Path

from pydantic import BaseModel, ConfigDict

from threedscriptors.data_handling.dataset.tasks import ElementSet, TaskSet


class DatasetCreationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    relaxation_tolerance: float | None = None
    relaxation_steps: int | None = None
    N_structures: int | None = None
    N_sampled_conformers: int = 1
    max_embed_attempts: int = 500
    max_MMFF_steps: int = 500

    # Molecule-standardization toggles applied before filter_mol in the
    # benchmark generators. Defaults strip common counter-ions and
    # neutralize formal charges so multi-fragment salt rows (HCl / Na+ / ...)
    # survive ingest as their neutral drug form instead of being rejected
    # outright by the single-fragment gate.
    strip_salts: bool = True
    neutralize: bool = True
    element_set: ElementSet = ElementSet.mace_off


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_chunk: int = 8192
    molecule_chunk: int = 4_096

    # zarr v3 sharding: one shard file groups this many chunks along the
    # growable (first) axis, so a shard = chunks_per_shard * chunk by
    # construction (zarr requires the shard shape to be a multiple of the
    # chunk shape). Small chunks keep training random-reads cheap; large
    # shards keep the on-disk file (inode) count tiny instead of one file
    # per chunk.
    atom_chunks_per_shard: int = 64
    molecule_chunks_per_shard: int = 256

    contains_smiles: bool = True
    tasks: TaskSet | None = None
