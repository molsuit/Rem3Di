from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    total_steps: int | None = None
    masking_probability: float | None = None
