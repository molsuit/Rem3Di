from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float | None = None
    mace_model_path: str
    train_dataset_path: str
    validation_dataset_path: Path
    model_dir: str
    training_data_dir: Path
    total_steps: int | None = None
    masking_probability: float | None = None
    wandb_active: bool = False
    normalized_targets: bool = True
    normalized_atomic_descriptors: bool = False


class TrainingMetadata(BaseModel):
    dataset_dir: str
    model_dir: str
    timestamp: datetime
