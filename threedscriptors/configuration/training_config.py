from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SplitStrategy(str, Enum):
    SINGLE = "single"  # hold‑out / train–val split
    REPEATED_CV = "repeated_cv"  # repeated k‑fold CV
    SCAFFOLD = "scaffold" # Bemis murcko scaffold split


@dataclass
class SplitConfig:
    """Configure one of the supported splitting strategies."""

    strategy: SplitStrategy
    train_val_ratios: tuple[float, float] = (0.8, 0.2)
    N_folds: int | None = None
    N_repeats: int | None = None
    shuffle: bool = True


class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    training_name: str
    run_group: str | None = None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float | None = None
    noise_level: float | None = None
    split_config: SplitConfig
    mace_model_path: Path
    dataset_path: Path
    model_config_path: Path
    total_steps: int | None = None
    wandb_active: bool = False


class TrainingMetadata(BaseModel):
    dataset_dir: Path
    model_dir: Path
    timestamp: datetime

