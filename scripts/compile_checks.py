"""GPU checks for torch.compile of the pretraining encoder and decoder.

Subcommands (run from the repository root with ``uv run python``):

``torch-repro``
    The PyTorch bug alone: ``nn.Linear`` on a ``(molecules, atoms, atoms,
    channels)`` tensor under ``torch.compile(dynamic=True)`` traces the last atom
    axis as ``(atoms**2)//atoms`` and inductor's triton backward fails with
    ``CantSplit``. Runs the same pair transitions with ``PairLinear``, which must
    compile. Exits non-zero only if a ``PairLinear`` variant fails.
``model-repro``
    The drug_like_small encoder and decoder (random inputs with the
    preprocessor's shapes, training mode, dropout on), compiled with dynamic
    shapes, forward and backward over several padded batch shapes; fails on a
    second graph per module.
``smoke-configs``
    Writes the compile-on and compile-off training configs of the pretraining
    smoke runs (one epoch, invariant statistics from 50 batches, probe off).
``summarise``
    Per-window throughput, recompiles and losses of the smoke runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from pydantic import TypeAdapter
from torch import nn

from remedi.configuration.architecture_config import (
    ArchitectureConfig,
    EncoderDecoderArchitectureConfig,
)
from remedi.model.pair_linear import PairLinear
from remedi.training.data.sample import PreprocessedSample

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DRUG_LIKE_SMALL = REPOSITORY_ROOT / "configs/training/drug_like_small"
#: The log line torch writes for every recompilation with
#: ``torch._logging.set_logs(recompiles=True)``.
RECOMPILE_MARKER = "Recompiling function"


# --- torch-repro --------------------------------------------------------------


class PairTransitionStack(nn.Module):
    """Two pre-norm residual MLPs with dropout on a pair tensor, built from
    ``linear_class`` (``nn.Linear`` reproduces the bug)."""

    def __init__(self, linear_class: type[nn.Linear], channels: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(channels),
                linear_class(channels, 2 * channels),
                nn.SiLU(),
                linear_class(2 * channels, channels),
                nn.Dropout(0.1),
            )
            for _ in range(2)
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            pair = pair + layer(pair)
        return pair


def _compiles(module: nn.Module, device: str) -> str | None:
    """None when forward and backward compile and run for two shapes, else
    the first line of the error."""
    torch._dynamo.reset()
    compiled = torch.compile(module.to(device).train(), dynamic=True)
    try:
        for molecules, atoms in [(16, 32), (8, 41)]:
            pair = torch.randn(
                molecules, atoms, atoms, 64, device=device, requires_grad=True
            )
            compiled(pair).square().mean().backward()
    except Exception as error:  # reported, not handled
        return f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    return None


def torch_repro(device: str) -> int:
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    failures = 0
    for linear_class in (nn.Linear, PairLinear):
        error = _compiles(PairTransitionStack(linear_class), device)
        expectation = (
            "the PyTorch bug" if linear_class is nn.Linear else "the fix, must pass"
        )
        print(f"  {linear_class.__name__:10s} ({expectation}): {error or 'OK'}")
        if error is not None and linear_class is PairLinear:
            failures += 1
    return failures


# --- model-repro --------------------------------------------------------------


def _architecture() -> EncoderDecoderArchitectureConfig:
    raw = yaml.safe_load((DRUG_LIKE_SMALL / "architecture_config.yaml").read_text())
    raw.pop("mace_config")  # the encoder and decoder do not need MACE
    config = TypeAdapter(ArchitectureConfig).validate_python(raw)
    if not isinstance(config, EncoderDecoderArchitectureConfig):
        raise TypeError("drug_like_small is expected to be an encoder_decoder config")
    return config


def _sample(
    config: EncoderDecoderArchitectureConfig, molecules: int, atoms: int, device: str
) -> PreprocessedSample:
    dimensions = config.dimensions
    lengths = torch.randint(atoms // 2, atoms + 1, (molecules,))
    lengths[0] = atoms
    padding_mask = (torch.arange(atoms)[None, :] >= lengths[:, None]).to(device)
    pair_mask = ~(padding_mask[:, :, None] | padding_mask[:, None, :])
    pair_shape = (molecules, atoms, atoms)
    return PreprocessedSample(
        preprocessed_atomic_embeddings=torch.randn(
            molecules, atoms, dimensions.atomic_descriptor_dimension, device=device
        ),
        padding_mask=padding_mask,
        initial_pair_representation=torch.randn(
            *pair_shape, dimensions.pair_dimension, device=device
        )
        * pair_mask[..., None],
        geometrical_encoding=torch.rand(
            *pair_shape, dimensions.radial_basis_dimension, device=device
        ),
        pair_mask=pair_mask,
    )


def model_repro(device: str) -> int:
    torch.manual_seed(0)
    config = _architecture()
    dimensions = config.dimensions
    encoder = config.encoder_config.build(
        config.global_aggregator_config.build(dimensions), dimensions
    )
    decoder = config.decoder_config.build(dimensions)
    compiled_encoder = torch.compile(encoder.float().to(device).train(), dynamic=True)
    compiled_decoder = torch.compile(decoder.float().to(device).train(), dynamic=True)
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(
            recompile_limit=1, fail_on_recompile_limit_hit=True
        ):
            for molecules, atoms in [(16, 32), (24, 40), (8, 56), (16, 37)]:
                start = time.perf_counter()
                sample = _sample(config, molecules, atoms, device)
                noise_level = torch.rand(molecules, device=device) + 0.1
                descriptor = compiled_encoder(sample)
                output = compiled_decoder(sample, descriptor, noise_level)
                output.square().mean().backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                print(
                    f"  batch ({molecules} molecules, {atoms} atoms): forward and "
                    f"backward in {time.perf_counter() - start:.1f} s"
                )
    except Exception as error:  # reported, not handled
        print(f"  FAILED {type(error).__name__}: {str(error).splitlines()[0][:300]}")
        return 1
    print("  OK: one graph per module for every batch shape")
    return 0


# --- smoke-configs ------------------------------------------------------------


def smoke_configs(
    base_config: Path,
    output_base: Path,
    configs_directory: Path,
    dataset_root: Path | None,
    mace_model_path: Path | None,
) -> int:
    """Writes ``compile_on.yaml`` and ``compile_off.yaml`` (and, with
    ``mace_model_path``, the architecture config they point to) into
    ``configs_directory``."""
    configs_directory.mkdir(parents=True, exist_ok=True)
    architecture = yaml.safe_load(
        (DRUG_LIKE_SMALL / "architecture_config.yaml").read_text()
    )
    if mace_model_path is not None:
        architecture["mace_config"]["model_path"] = str(mace_model_path.resolve())
    architecture_path = configs_directory / "architecture_config.yaml"
    architecture_path.write_text(yaml.safe_dump(architecture, sort_keys=False))
    model_path = Path(architecture["mace_config"]["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MACE model {model_path} not found: set MACE_MODEL_PATH"
        )

    training = yaml.safe_load(base_config.read_text())
    if dataset_root is not None:
        for component in training["data"]["components"]:
            component["dataset"] = str(
                dataset_root.resolve() / Path(component["dataset"]).name
            )
    for component in training["data"]["components"]:
        if not Path(component["dataset"]).is_dir():
            raise FileNotFoundError(
                f"dataset {component['dataset']} not found: set DATASET_ROOT"
            )
    training.update(
        epochs=1,
        normalization_batches=50,
        model_config_path=str(architecture_path.resolve()),
        output_base=str(output_base.resolve()),
        run_group="compile_smoke",
        training_directory=None,
        wandb_active=False,
    )
    training["probe"]["enabled"] = False
    for enabled in (True, False):
        name = "compile_on" if enabled else "compile_off"
        training["compile"]["enabled"] = enabled
        training["training_name"] = name
        path = configs_directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(training, sort_keys=False))
        print(path)
    return 0


# --- summarise ----------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    name: str
    windows: list[dict]
    recompiles: int
    losses: dict


def _run_summary(name: str, run_directory: Path, log_file: Path) -> RunSummary:
    throughput_file = run_directory / "throughput.jsonl"
    windows = (
        [json.loads(line) for line in throughput_file.read_text().splitlines()]
        if throughput_file.is_file()
        else []
    )
    log_text = log_file.read_text(errors="replace") if log_file.is_file() else ""
    losses_file = run_directory / "training_losses.yaml"
    epochs = yaml.safe_load(losses_file.read_text()) if losses_file.is_file() else []
    return RunSummary(
        name=name,
        windows=windows,
        recompiles=log_text.count(RECOMPILE_MARKER),
        losses=epochs[-1] if epochs else {},
    )


def _format(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summarise(runs: list[tuple[str, Path, Path]]) -> int:
    summaries = [_run_summary(*run) for run in runs]
    print("\nPer 50-step window (rest = step - preprocessor: encoder + decoder")
    print("forward/backward and the optimizer)")
    columns: list[tuple[str, Callable[[dict], float]]] = [
        ("step_ms", lambda window: window["step_ms"]),
        ("preprocessor_ms", lambda window: window["preprocessor_ms"]),
        ("featurizer_ms", lambda window: window["featurizer_ms"]),
        ("rest_ms", lambda window: window["step_ms"] - window["preprocessor_ms"]),
    ]
    labels = [
        f"{summary.name}:{column}" for column, _ in columns for summary in summaries
    ]
    widths = [len(label) + 2 for label in labels]
    print(
        f"{'step':>6}"
        + "".join(
            f"{label:>{width}}" for label, width in zip(labels, widths, strict=True)
        )
    )
    steps = sorted(
        {window["global_step"] for summary in summaries for window in summary.windows}
    )
    for step in steps:
        cells = []
        for _, value in columns:
            for summary in summaries:
                window = next(
                    (w for w in summary.windows if w["global_step"] == step), None
                )
                cells.append(_format(None if window is None else value(window)))
        print(
            f"{step:>6}"
            + "".join(
                f"{cell:>{width}}" for cell, width in zip(cells, widths, strict=True)
            )
        )

    print(
        f"\n{'run':<14}{'recompiles':>12}{'train loss':>12}{'val loss':>12}"
        f"{'peak GiB':>10}"
    )
    for summary in summaries:
        peak = max(
            (window["gpu_peak_memory_gib"] for window in summary.windows),
            default=None,
        )
        print(
            f"{summary.name:<14}{summary.recompiles:>12}"
            f"{_format(summary.losses.get('denoising_train_loss'), 4):>12}"
            f"{_format(summary.losses.get('denoising_val_loss'), 4):>12}"
            f"{_format(peak, 2):>10}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("torch-repro", "model-repro"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cuda")
    configs = commands.add_parser("smoke-configs")
    configs.add_argument(
        "--base-config",
        type=Path,
        default=DRUG_LIKE_SMALL / "training_config.yaml",
    )
    configs.add_argument("--output-base", type=Path, required=True)
    configs.add_argument("--configs-directory", type=Path, required=True)
    configs.add_argument("--dataset-root", type=Path, default=None)
    configs.add_argument("--mace-model-path", type=Path, default=None)
    summary = commands.add_parser("summarise")
    summary.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("NAME", "RUN_DIRECTORY", "LOG_FILE"),
        required=True,
    )
    arguments = parser.parse_args()

    if arguments.command == "torch-repro":
        return torch_repro(arguments.device)
    if arguments.command == "model-repro":
        return model_repro(arguments.device)
    if arguments.command == "smoke-configs":
        return smoke_configs(
            arguments.base_config,
            arguments.output_base,
            arguments.configs_directory,
            arguments.dataset_root,
            arguments.mace_model_path,
        )
    return summarise(
        [
            (name, Path(run_directory), Path(log_file))
            for name, run_directory, log_file in arguments.run
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
