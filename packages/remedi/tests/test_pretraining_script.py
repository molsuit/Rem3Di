"""The pretraining script end to end, on the CPU with MACE-OFF23 medium.

Runs ``scripts/run_online_embedding_denoising_pretraining.py`` on a twelve
molecule corpus: the invariant statistics are estimated, an epoch trains, the
checkpoint and the training state are written, and a second launch on the same
run directory resumes instead of starting over.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

from remedi.embedding.calculator import RemediCalculator
from remedi.model.checkpoint import (
    ARCHITECTURE_FILENAME,
    ATOMIC_PREPROCESSOR_FILENAME,
    ENCODER_FILENAME,
)
from remedi.training.training_state import TRAINING_STATE_FILENAME
from remedi_data.yaml_io import read_yaml
from testing import embed_frame, write_small_dataset

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "run_online_embedding_denoising_pretraining.py"
)
MACE_OFF_MEDIUM = Path.home() / ".cache" / "mace" / "MACE-OFF23_medium.model"
SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "OC",
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCCCCCC",
    "CCCCCCCC",
]


def _architecture(update_pair: bool) -> dict:
    attention = {
        "num_heads": 2,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "update_pair": update_pair,
    }
    return {
        "kind": "encoder_decoder",
        "mace_config": {
            "model_path": str(MACE_OFF_MEDIUM),
            "device": "cpu",
            "dtype": "float32",
            "enable_cueq": False,
            "compile": False,
        },
        "embedding_preprocess_config": {
            "pseudoscalars": False,
            "pseudoscalar_dimension": 0,
            "chiral_embedding_dimension": 0,
            "mace_layer_indices": [1],
        },
        "encoder_config": {
            "N_layers": 2,
            "embedding_dimension": 32,
            "attention_layer_config": attention,
        },
        "decoder_config": {
            "N_layers": 2,
            "embedding_dimension": 16,
            "attention_layer_config": attention,
        },
        "global_aggregator_config": {
            "aggregator_config": {"kind": "attention", "num_heads": 2},
            "output_dim": 8,
        },
        "positional_encoding_config": {
            "N_radial_basis_functions": 16,
            "distance_cutoff": 8.0,
            "d_projection": 8,
        },
    }


def _write_configs(tmp_path: Path, *, epochs: int, update_pair: bool = True) -> Path:
    dataset = write_small_dataset(
        tmp_path,
        dataset_id="corpus",
        labels=[],
        metrics=None,
        targets=None,
        smiles=SMILES,
    )
    architecture_path = tmp_path / "architecture_config.yaml"
    architecture_path.write_text(yaml.safe_dump(_architecture(update_pair)))
    training_path = tmp_path / "training_config.yaml"
    training_path.write_text(
        yaml.safe_dump(
            {
                "training_name": "smoke",
                "dataloader": {
                    "batch_sampling": {
                        "kind": "bucket_batch",
                        "max_atoms_per_batch": 40,
                    },
                    "num_workers": 0,
                },
                "epochs": epochs,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "model_config_path": str(architecture_path),
                "output_base": str(tmp_path / "runs"),
                "training_directory": str(tmp_path / "runs" / "smoke"),
                "compile": {"enabled": False},
                "normalization_batches": 3,
                "data": {
                    "domain": "drug_like",
                    "components": [{"dataset": str(dataset)}],
                    "validation": {"kind": "molecule_fraction", "fraction": 0.3},
                },
            }
        )
    )
    return training_path


class _JobKilled(Exception):
    """Stands in for SLURM ending the job at its time limit."""


def _run_script(
    training_config: Path, monkeypatch, *, kill_after_epoch: int | None = None
) -> None:
    """Run the script's ``main``; ``kill_after_epoch`` ends it (with
    :class:`_JobKilled`) right after that epoch's training state is saved."""
    specification = importlib.util.spec_from_file_location("pretraining", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--training_config", str(training_config)]
    )
    specification.loader.exec_module(module)
    if kill_after_epoch is not None:
        save_training_state = module.save_training_state

        def save_then_kill(directory, *, completed_epochs, **state):
            save_training_state(directory, completed_epochs=completed_epochs, **state)
            if completed_epochs == kill_after_epoch:
                raise _JobKilled

        module.save_training_state = save_then_kill
    try:
        module.main()
    finally:
        # The script logs to a file through a module-level logger: release it
        # so the next launch in this process opens the log again.
        logger = logging.getLogger("remedi")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


@pytest.fixture
def mace_off_medium() -> Path:
    if not MACE_OFF_MEDIUM.exists():
        pytest.skip(f"{MACE_OFF_MEDIUM} is missing")
    return MACE_OFF_MEDIUM


@pytest.mark.usefixtures("mace_off_medium")
@pytest.mark.parametrize("update_pair", [True, False])
def test_the_script_trains_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch, update_pair: bool
) -> None:
    run_directory = tmp_path / "runs" / "smoke"
    training_config = _write_configs(tmp_path, epochs=2, update_pair=update_pair)

    with pytest.raises(_JobKilled):
        _run_script(training_config, monkeypatch, kill_after_epoch=1)

    losses = read_yaml(run_directory / "training_losses.yaml")
    assert [record["epoch"] for record in losses] == [1]
    first = losses[0]
    # 1.0 is the zero-output decoder; one short epoch stays close to it.
    assert math.isfinite(first["denoising_train_loss"])
    assert 0.2 < first["denoising_val_loss"] < 1.5
    assert first["descriptor_standard_deviation"] > 0
    assert first["descriptor_effective_rank"] >= 1.0
    for filename in (ARCHITECTURE_FILENAME, ENCODER_FILENAME, TRAINING_STATE_FILENAME):
        assert (run_directory / filename).is_file()

    # The statistics were estimated from the data and travel with the checkpoint.
    preprocessor_state = torch.load(
        run_directory / ATOMIC_PREPROCESSOR_FILENAME, weights_only=True
    )
    assert bool(preprocessor_state["invariant_normalization.stats_set"])
    assert preprocessor_state["invariant_normalization.std"].ne(1.0).any()
    assert preprocessor_state["invariant_normalization.mean"].ne(0.0).any()

    state = torch.load(run_directory / TRAINING_STATE_FILENAME, weights_only=False)
    assert state["completed_epochs"] == 1
    assert set(state["modules"]) == {
        "encoder",
        "decoder",
        "atomic_preprocessor",
        "geometric_preprocessor",
    }
    first_epoch_steps = state["global_step"]
    assert first_epoch_steps > 0

    # The same config and run directory again: only epoch 2 is trained.
    _run_script(training_config, monkeypatch)

    losses = read_yaml(run_directory / "training_losses.yaml")
    assert [record["epoch"] for record in losses] == [1, 2]
    assert losses[0] == first
    state = torch.load(run_directory / TRAINING_STATE_FILENAME, weights_only=False)
    assert state["completed_epochs"] == 2
    assert state["global_step"] == 2 * first_epoch_steps
    assert "Resumed from" in (run_directory / "logfile.info").read_text()

    # The checkpoint embeds new molecules through the calculator.
    calculator = RemediCalculator(model_dir=run_directory, device="cpu")
    embedded = calculator.embed_atoms([embed_frame("CCO"), embed_frame("c1ccccc1")])
    assert embedded.shape == (2, 8)
    assert torch.isfinite(torch.from_numpy(embedded)).all()
    assert not torch.allclose(
        torch.from_numpy(embedded[0]), torch.from_numpy(embedded[1])
    )


@pytest.mark.usefixtures("mace_off_medium")
def test_a_run_does_not_resume_with_a_changed_schedule(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(_JobKilled):
        _run_script(_write_configs(tmp_path, epochs=2), monkeypatch, kill_after_epoch=1)
    with pytest.raises(ValueError, match="resumes only with the epochs"):
        _run_script(_write_configs(tmp_path, epochs=3), monkeypatch)
