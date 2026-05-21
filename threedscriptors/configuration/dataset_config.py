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
    # ETKDG retry budget per conformer. 200 is the validated production value
    # (see ``BenchmarkBuildConfig`` for the empirical justification); raising
    # it disproportionately inflates the wall-time tail on pathological mols.
    max_embed_attempts: int = 200
    # MMFF94 BFGS step cap per conformer.
    max_MMFF_steps: int = 100
    # MMFF94 non-bonded interaction cutoff in Å. RDKit's default (100.0)
    # already includes every atom pair for drug-sized molecules and is ~5x
    # cheaper per BFGS step than the previous 500.0 setting on large systems.
    mmff_non_bonded_thresh: float = 100.0

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
