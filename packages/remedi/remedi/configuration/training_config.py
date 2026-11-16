from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from remedi.configuration.dataloader_config import DataLoaderConfig
from remedi.training.data.mixture import PretrainingDataConfig


class CompileConfig(BaseModel):
    """torch.compile + shape-stability knobs."""

    enabled: bool = True
    # Round padded atom count up to this multiple to bound the inductor graph
    # cache. 1 = off; 8 / 16 are typical for FFN-heavy decoders. Pairs with
    # bucketed batching: with bucket_size choosing a narrow length range and
    # pad_multiple quantizing the max, inductor sees only a handful of shapes.
    pad_multiple: int = Field(default=1, gt=0)
    dynamo_cache_size_limit: int = Field(default=16, gt=0)


class VICRegConfig(BaseModel):
    """Optional VICReg variance + covariance regularizers on the descriptor.

    Defaults follow Bardes et al. 2022 (variance_weight=25, covariance_weight=1,
    target_std=1.0).
    """

    enabled: bool = False
    variance_weight: float = 25.0
    covariance_weight: float = 1.0
    target_std: float = 1.0


class ProbeConfig(BaseModel):
    """Frozen diagnostic linear probes on the molecular descriptor.

    Every ``every_n_steps`` training steps, a fixed set of ``n_probe_molecules``
    validation structures of the first mixture component is re-embedded (eval /
    no_grad) and a closed-form ridge is fit on a fixed fit/score split to measure
    how linearly decodable that dataset's labels (``dataset.yaml`` ``labels``,
    e.g. the PCQM4Mv2 HOMO-LUMO gap) are. The probe never backprops into the
    encoder.
    """

    enabled: bool = False
    every_n_steps: int = Field(default=500, gt=0)
    n_probe_molecules: int = Field(default=2048, gt=0)
    ridge_alpha: float = Field(default=1.0, gt=0)
    val_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    # None = probe every label of the dataset; else restrict to these names.
    properties: list[str] | None = None
    # Also log a rounded exact-match accuracy for integer-valued columns.
    report_count_accuracy: bool = False
    seed: int = 0


class ClassificationConfig(BaseModel):
    """Supervised single-label classification objective knobs.

    Only consumed by the classification trainer (a regression architecture whose
    head carries ``n_classes``); ignored by the self-supervised pretraining.
    """

    # Focal-loss focusing parameter; 0.0 -> (optionally weighted) cross-entropy.
    focal_gamma: float = 2.0
    # Up-weight rare classes via inverse-frequency focal alpha (sklearn
    # "balanced"), computed on the train split. Right default for the highly
    # imbalanced chiral-type task.
    class_balanced_alpha: bool = True


class TrainingRunConfig(BaseModel):
    """What every training run shares: optimisation, loading and outputs."""

    model_config = ConfigDict(extra="forbid")

    training_name: str
    run_group: str | None = None
    dataloader: DataLoaderConfig
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float | None = None
    model_config_path: Path
    output_base: Path
    training_directory: Path | None = None
    total_steps: int | None = None
    wandb_active: bool = False
    compile: CompileConfig = CompileConfig()


class PretrainingConfig(TrainingRunConfig):
    """Self-supervised denoising pretraining on a mixture of corpora."""

    data: PretrainingDataConfig
    noise_level: float
    vicreg: VICRegConfig = VICRegConfig()
    probe: ProbeConfig = ProbeConfig()


class ClassifierTrainingConfig(TrainingRunConfig):
    """Supervised training on one dataset's stored train / valid split."""

    dataset_path: Path
    classification: ClassificationConfig = ClassificationConfig()
