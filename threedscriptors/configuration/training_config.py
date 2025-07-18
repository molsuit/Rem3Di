from datetime import datetime
from pathlib import Path
from threedscriptors.training.dataset_splitting import SplitConfig
from pydantic import BaseModel
from enum import Enum

class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float | None = None
    mace_model_path: str
    split_config : SplitConfig
    dataset_path: str
    test_dataset_path: Path | None = None
    model_dir: str
    training_data_dir: Path
    total_steps: int | None = None
    noise_level: float | None = None
    wandb_active: bool = False
    normalized_targets: bool = True
    normalized_atomic_descriptors: bool = False


class TrainingMetadata(BaseModel):
    dataset_dir: str
    model_dir: str
    timestamp: datetime
