from pathlib import Path

from pydantic import BaseModel, ConfigDict

from threedscriptors.configuration.config_utils import IrrepType


class DatasetCreationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    relaxation_tolerance : float | None = None
    relaxation_steps : int | None = None
    N_structures: int | None = None
    N_sampled_conformers: int = 1
    max_embed_attempts: int = 500
    max_MMFF_steps: int = 500

class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_dim: int
    atom_chunk: int = 8192
    molecule_chunk: int = 4_096
    contains_smiles: bool = True
    irreps: IrrepType


