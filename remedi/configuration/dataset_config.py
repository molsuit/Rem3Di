from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.chemistry.conformers import ConformerEmbeddingConfig
from remedi.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from remedi.data_handling.physchem import DEFAULT_DESCRIPTORS


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
    N_structures: int | None = None
    conformers: ConformerEmbeddingConfig = Field(
        default_factory=ConformerEmbeddingConfig
    )


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
