from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.dataset.tasks import (
    ElementSet,
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from remedi.data_handling.physchem import DEFAULT_DESCRIPTORS


class FilterMoleculeStageConfig(BaseModel):
    """Knobs for the SMILES-side filter stage.

    Owned end-to-end by ``FilterMoleculeStage``: parse → standardize → filter
    → canonicalize → dedupe. Generators yield raw SMILES; the stage produces
    the clean ``SmilesData`` the rest of the pipeline consumes.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["smiles_filter"] = "smiles_filter"

    max_atoms: int | None = 100
    element_set: ElementSet = ElementSet.mace_off

    allow_charged: bool = True
    allow_radicals: bool = True
    allow_isotopes: bool = False
    allow_multifragment: bool = False

    strip_salts: bool = True
    neutralize: bool = True

    dedupe: bool = True


class FilterAtomsStageConfig(BaseModel):
    """Knobs for the Atoms-side filter stage (XYZ / tmQM / SDF sources).

    Operates on ``ase.Atoms`` directly: structures arriving from extxyz / SDF
    already carry coordinates, so this stage just checks size + element +
    hydrogen-coverage gates without re-parsing SMILES.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["atoms_filter"] = "atoms_filter"

    max_atoms: int | None = None
    # None means "do not check elements" — useful when the source is curated
    # (e.g. tmQM) and would otherwise reject every transition-metal complex.
    element_set: ElementSet | None = None

    reject_zero_h: bool = False
    min_h_heavy_ratio: float = 0.0
    # Drop structures with any pair of atoms closer than this (Angstrom). Guards
    # against degenerate geometries (e.g. overlapping / origin-placed atoms) that
    # make MACE divide by a ~zero interatomic distance and emit NaN embeddings.
    # None disables the check; ~0.5 A is safely below any real bond length.
    min_interatomic_distance: float | None = None


FilterStageConfig = Annotated[
    FilterMoleculeStageConfig | FilterAtomsStageConfig,
    Field(discriminator="kind"),
]


class PhysicochemicalDescriptorStageConfig(BaseModel):
    """Compute cheap RDKit physicochemical descriptors as per-structure targets.

    Each descriptor becomes one ``targets_system`` column (declared via
    :meth:`to_task_set`); they serve as linear-probe labels during pretraining.
    2D descriptors come from the SMILES, 3D ones (SASA) from the conformer — see
    ``remedi.data_handling.physchem``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["physchem_descriptors"] = "physchem_descriptors"
    descriptor_names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DESCRIPTORS)
    )

    def to_task_set(self) -> TaskSet:
        """Build a system-scope regression TaskSet, one column per descriptor."""
        return TaskSet.from_list(
            [
                TaskConfig(
                    name=name,
                    task_type=TaskType.regression,
                    scope=TaskScope.system,
                )
                for name in self.descriptor_names
            ]
        )


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
