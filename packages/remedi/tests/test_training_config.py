"""Training configuration: strict validation, defaults and the run directory."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import BaseModel, ValidationError

from remedi.configuration.dataloader_config import (
    BucketBatchSamplingConfig,
    DataLoaderConfig,
    RandomShuffleSamplingConfig,
)
from remedi.configuration.training_config import (
    ClassificationConfig,
    CompileConfig,
    FixedNoiseLevelConfig,
    LogNormalNoiseLevelConfig,
    PretrainingConfig,
    ProbeConfig,
)
from remedi.training.data.mixture import (
    DatasetComponent,
    MoleculeFractionValidation,
    PretrainingDataConfig,
)


def _pretraining_config(tmp_path: Path, **overrides) -> PretrainingConfig:
    values = {
        "training_name": "run",
        "dataloader": DataLoaderConfig(
            batch_sampling=BucketBatchSamplingConfig(max_atoms_per_batch=100)
        ),
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "model_config_path": tmp_path / "architecture.yaml",
        "output_base": tmp_path / "runs",
        "data": PretrainingDataConfig(
            domain="drug_like",
            components=[DatasetComponent(dataset=tmp_path / "corpus")],
            validation=MoleculeFractionValidation(fraction=0.1),
        ),
    }
    values.update(overrides)
    return PretrainingConfig.model_validate(values)


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (ProbeConfig, {"every_n_step": 10}),
        (CompileConfig, {"enable": False}),
        (LogNormalNoiseLevelConfig, {"mean": 1.0}),
        (FixedNoiseLevelConfig, {"level": 0.3, "noise_level": 0.3}),
        (ClassificationConfig, {"focal_gama": 1.0}),
        (RandomShuffleSamplingConfig, {"batch_size": 2, "shufle": True}),
        (BucketBatchSamplingConfig, {"max_atoms_per_batch": 10, "bucket_size": 4}),
        (
            DataLoaderConfig,
            {
                "batch_sampling": {"kind": "random_shuffle", "batch_size": 2},
                "workers": 2,
            },
        ),
    ],
)
def test_misspelt_options_are_rejected(
    model: type[BaseModel], values: dict[str, object]
) -> None:
    """A typo must fail instead of silently running with the default."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(values)


def test_total_steps_is_not_an_option(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="total_steps"):
        _pretraining_config(tmp_path, total_steps=10)


def test_gradient_clipping_has_a_usable_default(tmp_path: Path) -> None:
    """``clip_grad_norm_`` needs a number: the default must be one."""
    config = _pretraining_config(tmp_path)
    assert isinstance(config.max_grad_norm, float)
    assert config.max_grad_norm > 0
    with pytest.raises(ValidationError):
        _pretraining_config(tmp_path, max_grad_norm=None)


def test_the_seed_is_a_training_option(tmp_path: Path) -> None:
    assert _pretraining_config(tmp_path).seed == 0
    assert _pretraining_config(tmp_path, seed=7).seed == 7


def test_the_run_directory_is_numbered_and_named(tmp_path: Path) -> None:
    config = _pretraining_config(tmp_path, training_name="denoise")
    first = config.create_run_directory()
    second = config.create_run_directory()
    assert first.is_dir() and second.is_dir()
    assert first.parent == second.parent == tmp_path / "runs"
    assert first.name.startswith("0-") and first.name.endswith("-denoise")
    assert second.name.startswith("1-") and second.name.endswith("-denoise")


def test_an_explicit_training_directory_is_used(tmp_path: Path) -> None:
    directory = tmp_path / "snapshots" / "run"
    config = _pretraining_config(tmp_path, training_directory=directory)
    assert config.create_run_directory() == directory
    assert directory.is_dir()
    assert config.create_run_directory() == directory


def test_the_noise_level_is_log_normal_by_default(tmp_path: Path) -> None:
    assert isinstance(_pretraining_config(tmp_path).noise, LogNormalNoiseLevelConfig)


def test_the_noise_kind_is_chosen_in_the_config(tmp_path: Path) -> None:
    config = _pretraining_config(tmp_path, noise={"kind": "fixed", "level": 0.3})
    assert config.noise == FixedNoiseLevelConfig(level=0.3)
    with pytest.raises(ValidationError):
        _pretraining_config(tmp_path, noise={"kind": "fixed", "level": 0.0})


def test_vicreg_and_the_single_noise_level_are_gone(tmp_path: Path) -> None:
    for removed in ({"vicreg": {"enabled": False}}, {"noise_level": 0.3}):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _pretraining_config(tmp_path, **removed)


def test_fixed_noise_levels_are_all_the_level() -> None:
    levels = FixedNoiseLevelConfig(level=0.3).sample(4, device="cpu")
    assert torch.equal(levels, torch.full((4,), 0.3))


def test_log_normal_noise_levels_follow_the_generator_and_the_median() -> None:
    noise = LogNormalNoiseLevelConfig(median=0.5, log_standard_deviation=1.2)

    def draw(seed: int) -> torch.Tensor:
        return noise.sample(
            20_000, device="cpu", generator=torch.Generator().manual_seed(seed)
        )

    levels = draw(0)
    assert torch.equal(levels, draw(0))
    assert not torch.equal(levels, draw(1))
    assert (levels > 0).all()
    assert levels.median().item() == pytest.approx(0.5, rel=0.05)
    assert levels.log().std().item() == pytest.approx(1.2, rel=0.05)
