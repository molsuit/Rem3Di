import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from remedi.configuration.dataloader_config import DataLoaderConfig
from remedi.training.data.mixture import PretrainingDataConfig
from remedi.training.optimization import PlannedLearningRateSchedule, WarmUpThenDecay


class CompileConfig(BaseModel):
    """torch.compile + shape-stability knobs."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Round padded atom count up to this multiple to bound the inductor graph
    # cache. 1 = off; 8 / 16 are typical for FFN-heavy decoders. Pairs with
    # bucketed batching: with every batch drawn from a narrow length range and
    # pad_multiple quantizing the max, inductor sees only a handful of shapes.
    pad_multiple: int = Field(default=1, gt=0)
    dynamo_cache_size_limit: int = Field(default=16, gt=0)


class FixedNoiseLevelConfig(BaseModel):
    """The same noise level for every molecule."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["fixed"] = "fixed"
    level: float = Field(gt=0)

    def sample(
        self,
        count: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``(count,)`` noise levels; ``generator`` is unused (nothing is drawn)."""
        return torch.full((count,), self.level, device=device)


class LogNormalNoiseLevelConfig(BaseModel):
    """A log-normal noise level, drawn per molecule.

    Levels are in units of the standardised descriptors' deviation (1). The
    defaults put 95 % of the levels between about 0.05 and 5. Note that the
    512 noised channels still identify an atom's element at level 10, so these
    defaults leave the decoder little reason to use the molecular descriptor
    (decoder ablation, 2026-11-18).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["log_normal"] = "log_normal"
    median: float = Field(default=0.5, gt=0)
    log_standard_deviation: float = Field(default=1.2, gt=0)

    def sample(
        self,
        count: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``(count,)`` noise levels; ``generator`` (on ``device``) fixes them."""
        standard_normal = torch.randn(count, generator=generator, device=device)
        return torch.exp(
            math.log(self.median) + self.log_standard_deviation * standard_normal
        )


class LogUniformNoiseLevelConfig(BaseModel):
    """A noise level drawn per molecule, uniform in its logarithm between
    ``minimum`` and ``maximum``: every factor of ten in that range is drawn
    equally often."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["log_uniform"] = "log_uniform"
    minimum: float = Field(gt=0)
    maximum: float = Field(gt=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "LogUniformNoiseLevelConfig":
        if self.minimum >= self.maximum:
            raise ValueError(
                f"log_uniform noise needs minimum < maximum, got {self.minimum} "
                f"and {self.maximum}"
            )
        return self

    def sample(
        self,
        count: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``(count,)`` noise levels; ``generator`` (on ``device``) fixes them."""
        uniform = torch.rand(count, generator=generator, device=device)
        low, high = math.log(self.minimum), math.log(self.maximum)
        return torch.exp(low + (high - low) * uniform)


NoiseLevelConfig = Annotated[
    FixedNoiseLevelConfig | LogNormalNoiseLevelConfig | LogUniformNoiseLevelConfig,
    Field(discriminator="kind"),
]


class WarmUpCosineScheduleConfig(BaseModel):
    """Linear warm-up from ``learning_rate / warm_up_steps`` to ``learning_rate``
    over the first ``warm_up_fraction`` of the run's steps, then a cosine decay to
    ``final_learning_rate_fraction * learning_rate`` at the last step."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["warm_up_cosine"] = "warm_up_cosine"
    warm_up_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    final_learning_rate_fraction: float = Field(default=0.01, ge=0.0, le=1.0)

    def build(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> PlannedLearningRateSchedule:
        return PlannedLearningRateSchedule(
            optimizer,
            WarmUpThenDecay(
                total_steps=total_steps,
                warm_up_steps=round(self.warm_up_fraction * total_steps),
                cosine_decay=True,
                final_fraction=self.final_learning_rate_fraction,
            ),
        )


class ConstantScheduleConfig(BaseModel):
    """``learning_rate`` throughout, after an optional linear warm-up over the
    first ``warm_up_fraction`` of the run's steps."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["constant"] = "constant"
    warm_up_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)

    def build(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> PlannedLearningRateSchedule:
        return PlannedLearningRateSchedule(
            optimizer,
            WarmUpThenDecay(
                total_steps=total_steps,
                warm_up_steps=round(self.warm_up_fraction * total_steps),
                cosine_decay=False,
                final_fraction=1.0,
            ),
        )


LearningRateScheduleConfig = Annotated[
    WarmUpCosineScheduleConfig | ConstantScheduleConfig,
    Field(discriminator="kind"),
]


class ProbeConfig(BaseModel):
    """Frozen diagnostic linear probes on the molecular descriptor.

    Every ``every_n_steps`` training steps, a fixed set of ``n_probe_molecules``
    validation structures of the first mixture component is re-embedded (eval /
    no_grad) and a closed-form ridge is fit on a fixed fit/score split to measure
    how linearly decodable that dataset's labels (``dataset.yaml`` ``labels``,
    e.g. the PCQM4Mv2 HOMO-LUMO gap) are. The probe never backprops into the
    encoder.
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
    """The peak learning rate (after the warm-up)."""
    learning_rate_schedule: LearningRateScheduleConfig = WarmUpCosineScheduleConfig()
    weight_decay: float
    """AdamW weight decay on the weight matrices of linear layers; biases,
    normalisation scales, embeddings and pooling queries are not decayed."""
    max_grad_norm: float = Field(default=1.0, gt=0)
    """Gradients are clipped to this total norm every step."""
    seed: int = 0
    """Seeds torch, numpy and the data loaders (order and grouping of batches)."""
    normalization_batches: int = Field(default=200, gt=0)
    """Training batches the invariant normalisation statistics are estimated
    from, once, before the first epoch (precomputed normalisation only)."""
    model_config_path: Path
    output_base: Path
    training_directory: Path | None = None
    wandb_active: bool = False
    compile: CompileConfig = CompileConfig()

    def create_run_directory(self) -> Path:
        """Create and return the run's directory.

        ``training_directory`` when set (a snapshot made by ``submit_job``),
        otherwise a new ``{index}-{timestamp}-{training_name}`` directory under
        ``output_base``, numbered after the directories already there.
        """
        if self.training_directory is not None:
            self.training_directory.mkdir(parents=True, exist_ok=True)
            return self.training_directory
        self.output_base.mkdir(parents=True, exist_ok=True)
        index = len(list(self.output_base.glob("*/")))
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        directory = self.output_base / f"{index}-{timestamp}-{self.training_name}"
        directory.mkdir()
        return directory


class PretrainingConfig(TrainingRunConfig):
    """Self-supervised denoising pretraining on a mixture of corpora."""

    data: PretrainingDataConfig
    noise: NoiseLevelConfig = LogNormalNoiseLevelConfig()
    probe: ProbeConfig = ProbeConfig()


class ClassifierTrainingConfig(TrainingRunConfig):
    """Supervised training on one dataset's stored train / valid split."""

    dataset_path: Path
    classification: ClassificationConfig = ClassificationConfig()
