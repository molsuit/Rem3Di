from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    mace_model_path: str
    dataset_path: str
    total_steps: int | None = None
    masking_probability: float | None = None
    wandb_active: bool = False
