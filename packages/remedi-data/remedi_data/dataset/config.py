from pydantic import BaseModel, ConfigDict


class DatasetConfig(BaseModel):
    """``dataset_config.yaml``: the chunk and shard geometry of a dataset's zarr.

    Labels are not part of the zarr: their names and task types are in
    ``dataset.yaml`` (``DatasetSpec.labels``) and their values in
    ``table.parquet``.
    """

    model_config = ConfigDict(extra="forbid")

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
