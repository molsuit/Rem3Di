"""The Rem3Di checkpoint layout: which files a trained model directory holds.

A checkpoint directory holds the architecture yaml the model is rebuilt from
and one ``state_dict`` file per trained part::

    post_training_architecture_config.yaml   the architecture (written by the training scripts)
    encoder.pth                              the transformer encoder
    atomic_preprocessor.pth                  the atomic (MACE feature) preprocessor
    geometric_preprocessor.pth               the pair-geometry preprocessor
    classification_head.pth                  only for a model with heads (chiral classifier)

The frozen MACE foundation model is not saved; the architecture yaml names it.
:data:`DESCRIPTOR_IDENTITY_FILES` are the files that decide a descriptor
(``RemediConfig.identity`` hashes them together with the MACE model file).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

ARCHITECTURE_FILENAME = "post_training_architecture_config.yaml"
ENCODER_FILENAME = "encoder.pth"
ATOMIC_PREPROCESSOR_FILENAME = "atomic_preprocessor.pth"
GEOMETRIC_PREPROCESSOR_FILENAME = "geometric_preprocessor.pth"
CLASSIFICATION_HEAD_FILENAME = "classification_head.pth"

#: The weights every checkpoint holds: what the descriptor is computed from.
ENCODER_WEIGHT_FILES: tuple[str, ...] = (
    ENCODER_FILENAME,
    ATOMIC_PREPROCESSOR_FILENAME,
    GEOMETRIC_PREPROCESSOR_FILENAME,
)
#: The files whose contents decide the descriptor a checkpoint computes.
DESCRIPTOR_IDENTITY_FILES: tuple[str, ...] = (
    ARCHITECTURE_FILENAME,
    *ENCODER_WEIGHT_FILES,
)


def _module(owner: object, name: str) -> nn.Module:
    part = getattr(owner, name, None)
    if not isinstance(part, nn.Module):
        raise TypeError(f"{type(owner).__name__} has no {name!r} module to checkpoint")
    return part


def _parts(model: object) -> dict[str, nn.Module]:
    """The modules a checkpoint holds, by file name.

    ``model`` has an ``encoder`` and a ``preprocessor`` with an
    ``atomic_preprocessor`` and a ``geometric_preprocessor`` (a ``REM3DIModel``,
    a ``MultiTaskRegressionModel`` or the pretraining ``EncoderDecoderBundle``);
    a ``multitask_heads`` module, if present, is the classification head.
    """
    preprocessor = _module(model, "preprocessor")
    parts = {
        ENCODER_FILENAME: _module(model, "encoder"),
        ATOMIC_PREPROCESSOR_FILENAME: _module(preprocessor, "atomic_preprocessor"),
        GEOMETRIC_PREPROCESSOR_FILENAME: _module(
            preprocessor, "geometric_preprocessor"
        ),
    }
    if getattr(model, "multitask_heads", None) is not None:
        parts[CLASSIFICATION_HEAD_FILENAME] = _module(model, "multitask_heads")
    return parts


def save_checkpoint(model: object, directory: Path) -> None:
    """Write the weights of ``model`` into ``directory`` in the checkpoint layout.

    The architecture yaml is written by the caller, which knows the config.

    Raises:
        TypeError: if ``model`` lacks one of the parts (see :func:`_parts`).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, part in _parts(model).items():
        torch.save(part.state_dict(), directory / filename)


def load_checkpoint(
    model: object,
    directory: Path,
    map_location: str | torch.device | None = None,
) -> None:
    """Load the weights in ``directory`` into ``model`` (built from its architecture).

    Raises:
        FileNotFoundError: if a weight file the model needs is missing.
        RuntimeError: if a file's ``state_dict`` does not fit the model part.
        TypeError: if ``model`` lacks one of the parts (see :func:`_parts`).
    """
    directory = Path(directory)
    for filename, part in _parts(model).items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"{directory} has no {filename}")
        part.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True)
        )
