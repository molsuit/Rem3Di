from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    """``dataset_config.yaml``: the chunk and shard geometry of a dataset's zarr.

    The one description of the layout: the build and the structure preparation
    take it as their ``zarr`` setting and :meth:`MoleculeDataset.write` writes
    it beside the arrays. Labels are not part of the zarr: their names and task
    types are in ``dataset.yaml`` (``DatasetSpec.labels``) and their values in
    ``table.parquet``.
    """

    model_config = ConfigDict(extra="forbid")

    #: Atoms per chunk of ``positions`` and ``atomic_numbers``.
    atom_chunk: int = Field(default=8192, ge=1)
    #: Structures per chunk of the per-structure arrays and ``molecule_ptr``.
    molecule_chunk: int = Field(default=4096, ge=1)

    # zarr v3 sharding: one shard file groups this many chunks along the
    # first axis, so a shard = chunks_per_shard * chunk by construction (zarr
    # requires the shard shape to be a multiple of the chunk shape). Small
    # chunks keep training random-reads cheap; large shards keep the on-disk
    # file (inode) count tiny instead of one file per chunk.
    atom_chunks_per_shard: int = Field(default=64, ge=1)
    molecule_chunks_per_shard: int = Field(default=256, ge=1)
