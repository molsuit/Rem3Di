from datetime import datetime

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    mace_model_path: str
    dataset_path: str
    model_dir: str
    total_steps: int | None = None
    masking_probability: float | None = None
    wandb_active: bool = False


class TrainingMetadata(BaseModel):
    dataset_dir: str
    model_dir: str
    timestamp: datetime
