"""The checkpoint layout: one module saves and loads every Rem3Di model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from remedi.model.checkpoint import (
    CLASSIFICATION_HEAD_FILENAME,
    ENCODER_WEIGHT_FILES,
    load_checkpoint,
    save_checkpoint,
)


class _Preprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.atomic_preprocessor = nn.Linear(4, 3)
        self.geometric_preprocessor = nn.Linear(2, 5)


class _Model(nn.Module):
    def __init__(self, with_head: bool) -> None:
        super().__init__()
        self.preprocessor = _Preprocessor()
        self.encoder = nn.Linear(3, 6)
        self.multitask_heads = nn.Linear(6, 2) if with_head else None


def _parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("with_head", [False, True], ids=["encoder", "classifier"])
def test_a_saved_checkpoint_loads_back_into_a_fresh_model(
    tmp_path: Path, with_head: bool
) -> None:
    torch.manual_seed(0)
    trained = _Model(with_head)
    save_checkpoint(trained, tmp_path)

    expected_files = set(ENCODER_WEIGHT_FILES)
    if with_head:
        expected_files.add(CLASSIFICATION_HEAD_FILENAME)
    assert {path.name for path in tmp_path.iterdir()} == expected_files

    torch.manual_seed(1)
    fresh = _Model(with_head)
    load_checkpoint(fresh, tmp_path, map_location="cpu")
    loaded, saved = _parameters(fresh), _parameters(trained)
    assert loaded.keys() == saved.keys()
    for name, value in saved.items():
        torch.testing.assert_close(loaded[name], value)


def test_a_model_with_heads_needs_the_head_file(tmp_path: Path) -> None:
    save_checkpoint(_Model(with_head=False), tmp_path)
    with pytest.raises(FileNotFoundError, match=CLASSIFICATION_HEAD_FILENAME):
        load_checkpoint(_Model(with_head=True), tmp_path, map_location="cpu")
