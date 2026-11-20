"""Where the time of a pretraining step goes: a torch.profiler trace of the real
training step (``denoising_step``, backward, gradient clipping, optimizer) on a
training config's data and architecture.

Warms up (compilation, loader workers) for ``--warmup_steps``, then profiles
``--profile_steps`` steps and prints, per phase (forward, backward, clip,
optimizer), the host wall time and the GPU kernel time, the GPU's idle share of
the step, the host-device synchronisations, and the kernels with the most GPU
time. Writes a Chrome trace (``trace.json``, open in https://ui.perfetto.dev).

    uv run python scripts/profile_pretraining_step.py \\
        --training_config <training_config.yaml> --output_dir <dir> [--compile]
"""

from __future__ import annotations

import argparse
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TextIO

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from remedi.configuration.architecture_config import ArchitectureConfig
from remedi.configuration.training_config import PretrainingConfig
from remedi.model.preprocessing.invariant_statistics import (
    ensure_invariant_statistics,
)
from remedi.model.preprocessing.preprocessing import PreprocessorWithAtomicEmbedding
from remedi.training.data.mixture import resolve_mixture
from remedi.training.data.training_dataset import atoms_getitem
from remedi.training.optimization import parameter_groups
from remedi.training.pretraining import denoising_step
from remedi_data.yaml_io import read_model

PHASES = ("forward", "backward", "clip", "optimizer")
SYNCHRONISING_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the config's compile.enabled",
    )
    parser.add_argument("--warmup_steps", type=int, default=60)
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--top_kernels", type=int, default=25)
    parser.add_argument(
        "--find_synchronisations",
        action="store_true",
        help="after the warm-up, run two steps under torch.cuda sync debug mode and "
        "print every host-device synchronisation with the repository lines behind it",
    )
    parser.add_argument(
        "--matmul_precision",
        choices=["highest", "high"],
        default="highest",
        help="float32 matmuls: 'highest' (full FP32) or 'high' (TF32); the MACE "
        "featurizer refuses 'high' (it must run in full float32)",
    )
    parser.add_argument(
        "--autocast",
        choices=["none", "bfloat16"],
        default="none",
        help="run the step's forward under autocast (the MACE featurizer opts out)",
    )
    return parser.parse_args()


def endless(loader):
    while True:
        yield from loader


def training_config_with_overrides(arguments: argparse.Namespace) -> PretrainingConfig:
    """The training config with the command line's compile and worker overrides."""
    training_config = read_model(arguments.training_config, PretrainingConfig)
    if arguments.compile is not None:
        training_config.compile.enabled = arguments.compile
    if arguments.num_workers is not None:
        training_config.dataloader.num_workers = arguments.num_workers
        training_config.dataloader.persistent_workers = arguments.num_workers > 0
    return training_config


def report_synchronisations(step, steps: int) -> None:
    """Run ``steps`` steps under ``torch.cuda.set_sync_debug_mode("warn")`` and
    print each synchronisation, keyed by the repository frames of its stack."""
    repository = str(Path(__file__).resolve().parents[1])
    found: dict[tuple[str, ...], int] = defaultdict(int)

    def record(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        if "synchronizing" not in str(message):
            return
        stack = traceback.extract_stack()[:-1]
        in_repository = [
            frame
            for frame in stack
            if frame.filename.startswith(repository) and ".venv" not in frame.filename
        ]
        # The repository frames, then the innermost library frames below them.
        innermost = [frame for frame in stack[-3:] if frame not in in_repository]
        frames = tuple(
            f"{frame.filename.removeprefix(repository + '/')}:{frame.lineno} "
            f"({frame.name})"
            for frame in in_repository[-4:] + innermost
        )
        found[frames] += 1

    previous = warnings.showwarning
    # Same signature; ty compares the function names.
    warnings.showwarning = record  # ty: ignore[invalid-assignment]
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for _ in range(steps):
            step()
        torch.cuda.set_sync_debug_mode("default")
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = previous
    print(f"host-device synchronisations over {steps} steps:")
    for frames, count in sorted(found.items(), key=lambda item: -item[1]):
        print(f"  {count / steps:.1f} per step:")
        for frame in frames:
            print(f"      {frame.split('site-packages/')[-1]}")


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("profiling a training step needs a CUDA device")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision(arguments.matmul_precision)
    autocast = torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=arguments.autocast == "bfloat16"
    )

    training_config = training_config_with_overrides(arguments)
    architecture_config = read_model(
        training_config.model_config_path, ArchitectureConfig
    )
    torch.manual_seed(training_config.seed)

    data = resolve_mixture(training_config.data)
    datasets = data.open_datasets(atoms_getitem)
    loader = training_config.dataloader.build(
        datasets.training,
        lengths=datasets.training_atom_counts,
        collate_fn=architecture_config.molecule_collator(),
        shuffle=True,
        seed=training_config.seed,
    )

    bundle = architecture_config.build()
    preprocessor = bundle.preprocessor.to(device)
    encoder = bundle.encoder.to(device)
    decoder = bundle.decoder.to(device)
    compile_config = training_config.compile
    torch._dynamo.config.cache_size_limit = compile_config.dynamo_cache_size_limit
    if isinstance(preprocessor, PreprocessorWithAtomicEmbedding):
        preprocessor.pad_multiple = compile_config.pad_multiple
        if compile_config.enabled:
            preprocessor.mace_featurizer.compile_features()
    groups = parameter_groups((encoder, decoder, preprocessor))
    parameters = groups.all
    optimizer = torch.optim.AdamW(
        groups.for_optimizer(training_config.weight_decay),
        lr=training_config.learning_rate,
    )
    ensure_invariant_statistics(
        preprocessor,
        loader,
        max_batches=training_config.normalization_batches,
        device=device,
    )
    if compile_config.enabled:
        encoder = torch.compile(encoder, dynamic=True)
        decoder = torch.compile(decoder, dynamic=True)
    for module in (preprocessor, encoder, decoder):
        module.train()

    batches = endless(loader)

    def step() -> int:
        samples = next(batches)
        samples.to_(device)
        with record_function("forward"), autocast:
            result = denoising_step(
                samples,
                preprocessor=preprocessor,
                encoder=encoder,
                decoder=decoder,
                noise=training_config.noise,
            )
        with record_function("backward"):
            result.denoising_loss.backward()
        with record_function("clip"):
            torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=training_config.max_grad_norm
            )
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        return result.atom_count

    started = time.perf_counter()
    for _ in range(arguments.warmup_steps):
        step()
    torch.cuda.synchronize()
    print(
        f"warm-up: {arguments.warmup_steps} steps in "
        f"{time.perf_counter() - started:.1f} s (compile "
        f"{'on' if compile_config.enabled else 'off'}, matmul precision "
        f"{arguments.matmul_precision}, autocast {arguments.autocast})"
    )

    if arguments.find_synchronisations:
        report_synchronisations(step, steps=2)

    atoms = 0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as profiler:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(arguments.profile_steps):
            atoms += step()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
    profiler.export_chrome_trace(str(arguments.output_dir / "trace.json"))
    report(profiler, wall_seconds, atoms, arguments)


def report(profiler, wall_seconds: float, atoms: int, arguments) -> None:
    steps = arguments.profile_steps
    events = profiler.events()
    phase_host_us: dict[str, float] = defaultdict(float)
    kernel_us: dict[str, float] = defaultdict(float)
    kernel_count: dict[str, int] = defaultdict(int)
    synchronisations = 0
    for event in events:
        if event.name in PHASES and event.device_type == torch.autograd.DeviceType.CPU:
            phase_host_us[event.name] += event.cpu_time_total
        if event.name in SYNCHRONISING_CALLS:
            synchronisations += 1
        # Annotations (the phases, Optimizer.step) also appear as device events.
        is_annotation = event.name in PHASES or event.name.startswith("Optimizer.")
        if event.device_type == torch.autograd.DeviceType.CUDA and not is_annotation:
            kernel_us[event.name] += event.device_time_total
            kernel_count[event.name] += 1
    busy_ms = sum(kernel_us.values()) / 1000.0 / steps
    step_ms = 1000.0 * wall_seconds / steps
    lines = [
        f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}",
        f"step {step_ms:.1f} ms wall, {busy_ms:.1f} ms GPU kernels "
        f"(GPU idle {100.0 * max(0.0, 1.0 - busy_ms / step_ms):.0f} %), "
        f"{atoms / wall_seconds:,.0f} atoms/s, "
        f"{synchronisations / steps:.1f} host-device synchronisations per step",
        "",
        # Host time only: backward kernels are launched from autograd's own
        # thread, so they cannot be attributed to the phases on the device side.
        f"{'phase':<10} {'host ms':>9}",
    ]
    for phase in PHASES:
        lines.append(f"{phase:<10} {phase_host_us[phase] / 1000 / steps:>9.1f}")
    lines += ["", f"top {arguments.top_kernels} kernels by GPU time per step:"]
    ranked = sorted(kernel_us.items(), key=lambda item: -item[1])
    for name, microseconds in ranked[: arguments.top_kernels]:
        share = 100.0 * microseconds / max(sum(kernel_us.values()), 1.0)
        lines.append(
            f"{microseconds / 1000 / steps:>7.2f} ms {share:>5.1f} % "
            f"x{kernel_count[name] / steps:>5.1f}  {name[:110]}"
        )
    text = "\n".join(lines)
    print(text)
    (arguments.output_dir / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
