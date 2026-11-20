"""GPU checks of the neighbour-list MACE featurizer (driven by run_gpu_checks.sh).

Three subcommands:

``benchmark``
    MACE-POLAR node features per ~5250-atom batch, float32 + cuequivariance, of two
    batch sets of real pretraining structures: "small" (random PCQM4Mv2 molecules)
    and "large" (GEOM-drugs molecules of 60 to 120 atoms). ``--mode`` picks the
    graph: ``neighbor_*`` is the featurizer on the collated neighbour list,
    ``dense_*`` the same layers on every ordered pair of atoms within a molecule,
    with the pair index built inside the (compiled) graph from the padded shape:
    the featurizer before the neighbour list (commit ff2cd47). One JSON line per
    batch set goes to ``<output_dir>/timings.jsonl``, the features of each set's
    first batch (flat atom order) to ``features_<mode>_<set>.npy``.

``smoke-config``
    The compile-off pretraining smoke configuration: the repository's
    ``configs/training/drug_like_small`` with the given MACE model and datasets,
    one epoch, 50 normalisation batches, no probe. Writes
    ``<config_dir>/training_config.yaml`` and ``architecture_config.yaml``.

``summary``
    Tables of the benchmark (with each neighbour-list mode's feature difference to
    the dense one) and of a smoke run's ``throughput.jsonl`` next to the dense
    baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from remedi.configuration.mace_config import MaceConfig
from remedi.model.featurizer import MaceFeaturizer, pad_molecules
from remedi.training.data.sample import MoleculeCollator, Sample
from remedi.training.data.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SMOKE_TEMPLATE = REPOSITORY / "configs" / "training" / "drug_like_small"
ATOMS_PER_BATCH = 5250  # drug_like_small: max_atoms_per_batch
BATCHES_PER_SET = 4
PAD_MULTIPLE = 1  # drug_like_small: compile.pad_multiple
MODES = ("neighbor_eager", "neighbor_compiled", "dense_eager", "dense_compiled")
# The dense featurizer (eager) in the same smoke run on an H200 (DAIS job 929582):
# per 50-step window, steps 50 to 400.
DENSE_BASELINE_STEP_MS = (85, 207, 184, 135, 111, 112, 81, 85)
DENSE_BASELINE_FEATURIZER_MS = (23, 156, 132, 85, 60, 61, 34, 36)


# ---------------------------------------------------------------- benchmark


def dense_pair_index(molecules: int, atoms: int, device: torch.device) -> torch.Tensor:
    """Every ordered pair ``(sender, receiver)`` of distinct atoms within each
    molecule of a padded batch, as flat padded atom indices
    ``[2, molecules * atoms * (atoms - 1)]``: the dense graph of commit ff2cd47."""
    pair = torch.arange(atoms * (atoms - 1), device=device)
    sender_slot = pair // (atoms - 1)
    remainder = pair % (atoms - 1)
    receiver_slot = remainder + (remainder >= sender_slot).long()
    molecule_offset = torch.arange(molecules, device=device)[:, None] * atoms
    sender = (molecule_offset + sender_slot[None, :]).reshape(-1)
    receiver = (molecule_offset + receiver_slot[None, :]).reshape(-1)
    return torch.stack([sender, receiver])


def _batch_set(dataset_path: Path, smallest: int, largest: int, seed: int) -> list:
    """``BATCHES_PER_SET`` lists of molecules of ``smallest..largest`` atoms, about
    ATOMS_PER_BATCH atoms each."""
    dataset = TrainingMoleculeDataset(dataset_path, get_item=atoms_getitem).open()
    counts = dataset.atom_counts()
    eligible = np.flatnonzero((counts >= smallest) & (counts <= largest))
    if len(eligible) == 0:
        raise ValueError(f"{dataset_path} has no molecules of {smallest}..{largest}")
    generator = np.random.default_rng(seed)
    batches = []
    for _ in range(BATCHES_PER_SET):
        chosen, atoms = [], 0
        while atoms < ATOMS_PER_BATCH:
            index = int(eligible[generator.integers(len(eligible))])
            chosen.append(dataset[index])
            atoms += int(counts[index])
        batches.append(chosen)
    return batches


def _features_function(
    featurizer: MaceFeaturizer, kind: str, compiled: bool
) -> Callable[[Sample], torch.Tensor]:
    """The features ``(atoms, C)`` of a collated batch's real atoms, on the
    neighbour list (``kind`` "neighbor") or on every intra-molecule pair ("dense")."""
    if kind == "neighbor":
        if compiled:
            featurizer.compile_features()

        def neighbor_features(sample: Sample) -> torch.Tensor:
            padded = pad_molecules(sample, PAD_MULTIPLE)
            return featurizer(padded)[~padded.padding_mask]

        return neighbor_features

    def dense_graph_features(
        atomic_numbers: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        molecules, atoms = atomic_numbers.shape
        edge_index = dense_pair_index(molecules, atoms, positions.device)
        return featurizer.features(atomic_numbers, positions, edge_index)

    evaluate = (
        torch.compile(dense_graph_features, fullgraph=True, dynamic=True)
        if compiled
        else dense_graph_features
    )

    def dense_features(sample: Sample) -> torch.Tensor:
        padded = pad_molecules(sample, PAD_MULTIPLE)
        positions = padded.positions.to(featurizer.dtype)
        if compiled:
            for tensor in (padded.atomic_numbers, positions):
                torch._dynamo.maybe_mark_dynamic(tensor, 0)
                torch._dynamo.maybe_mark_dynamic(tensor, 1)
        with torch.no_grad():
            output = evaluate(padded.atomic_numbers, positions)
        return output[~padded.padding_mask]

    return dense_features


def _padded_shape(sample: Sample) -> tuple[int, int]:
    """``(molecules, longest molecule)`` of a collated batch."""
    if sample.padding_mask is None:
        raise ValueError("expected a collated batch with its padding mask")
    return int(sample.padding_mask.shape[0]), int(sample.padding_mask.shape[1])


def _batch_description(collated: list[Sample]) -> dict:
    shapes = [_padded_shape(sample) for sample in collated]
    return {
        "atoms_per_batch": [len(sample) for sample in collated],
        "molecules_per_batch": [molecules for molecules, _ in shapes],
        "longest_molecule": [atoms for _, atoms in shapes],
        "neighbor_pairs": [
            int(sample.edge_index.shape[1])
            for sample in collated
            if sample.edge_index is not None
        ],
        "dense_pairs": [
            molecules * max(atoms, 2) * (max(atoms, 2) - 1)
            for molecules, atoms in shapes
        ],
    }


def _time_set(
    features: Callable[[Sample], torch.Tensor],
    work: list[Sample],
    repeats: int,
    feature_path: Path,
) -> dict:
    """First call (compilation included), then ``repeats`` timed calls after a
    warm-up on every batch; saves the first batch's features."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    first = features(work[0])
    torch.cuda.synchronize()
    first_call = time.perf_counter() - started
    np.save(feature_path, first.float().cpu().numpy())
    # Warm-up: the other batch shapes (recompiles, if any, happen here).
    for sample in work[1:]:
        features(sample)
    torch.cuda.synchronize()
    graphs_after_warmup = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(repeats):
        sample = work[repeat % len(work)]
        torch.cuda.synchronize()
        started = time.perf_counter()
        features(sample)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return {
        "first_call_seconds": first_call,
        "median_ms": 1000.0 * float(np.median(times)),
        "min_ms": 1000.0 * float(np.min(times)),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "unique_graphs_after_warmup": graphs_after_warmup,
        "unique_graphs_after_timing": torch._dynamo.utils.counters["stats"][
            "unique_graphs"
        ],
    }


def benchmark(arguments: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the benchmark needs a CUDA device")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    kind, _, compilation = arguments.mode.partition("_")
    config = MaceConfig.model_validate(
        {
            "model_path": arguments.model_path,
            "device": "cuda",
            "dtype": "float32",
            "enable_cueq": True,
        }
    )
    features = _features_function(
        config.build_featurizer(), kind, compiled=compilation == "compiled"
    )
    collate = MoleculeCollator(cutoff=config.r_max)
    device = torch.device("cuda")
    sets = {
        "small": _batch_set(arguments.pcqm4mv2, 1, 10_000, seed=0),
        "large": _batch_set(arguments.geom_drugs, 60, 120, seed=1),
    }
    for set_name, molecule_lists in sets.items():
        started = time.perf_counter()
        collated = [collate(molecules) for molecules in molecule_lists]
        collate_seconds = (time.perf_counter() - started) / len(collated)
        work = [sample.to(device) for sample in collated]
        timing = _time_set(
            features,
            work,
            arguments.repeats,
            arguments.output_dir / f"features_{arguments.mode}_{set_name}.npy",
        )
        record = {
            "mode": arguments.mode,
            "set": set_name,
            "gpu": torch.cuda.get_device_name(),
            **_batch_description(collated),
            "collate_ms_cpu": 1000.0 * collate_seconds,
            **timing,
        }
        print(json.dumps(record), flush=True)
        with open(arguments.output_dir / "timings.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------- smoke config


def smoke_config(arguments: argparse.Namespace) -> None:
    from remedi.configuration.architecture_config import ArchitectureConfig
    from remedi.configuration.training_config import PretrainingConfig
    from remedi_data.yaml_io import read_model, read_yaml, write_yaml

    for path in (arguments.model_path, arguments.pcqm4mv2, arguments.geom_drugs):
        if not path.exists():
            raise FileNotFoundError(path)
    arguments.config_dir.mkdir(parents=True, exist_ok=True)

    architecture = read_yaml(SMOKE_TEMPLATE / "architecture_config.yaml")
    architecture["mace_config"]["model_path"] = str(arguments.model_path)
    architecture_path = arguments.config_dir / "architecture_config.yaml"
    write_yaml(architecture_path, architecture)

    training = read_yaml(SMOKE_TEMPLATE / "training_config.yaml")
    datasets = {"pcqm4mv2": arguments.pcqm4mv2, "geom_drugs": arguments.geom_drugs}
    for component in training["data"]["components"]:
        name = Path(component["dataset"]).name
        if name not in datasets:
            raise ValueError(f"the smoke template names an unknown dataset {name}")
        component["dataset"] = str(datasets[name])
    training["compile"]["enabled"] = False
    training["dataloader"]["num_workers"] = arguments.num_workers
    training["dataloader"]["persistent_workers"] = arguments.num_workers > 0
    if arguments.num_workers == 0:
        training["dataloader"].pop("prefetch_factor", None)
    training.update(
        epochs=1,
        normalization_batches=50,
        model_config_path=str(architecture_path),
        output_base=str(arguments.output_base),
        run_group="neighbor_list_smoke",
        training_name="compile_off",
        training_directory=None,
        wandb_active=False,
        probe={"enabled": False, "every_n_steps": 500},
    )
    training_path = arguments.config_dir / "training_config.yaml"
    write_yaml(training_path, training)
    # Fail here, not in the run, on a configuration that does not validate.
    read_model(architecture_path, ArchitectureConfig)
    read_model(training_path, PretrainingConfig)
    print(training_path)


# ---------------------------------------------------------------- summary


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header[column]), *(len(row[column]) for row in rows))
        for column in range(len(header))
    ]
    lines = [
        " | ".join(
            text.rjust(width) for text, width in zip(header, widths, strict=True)
        )
    ]
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            " | ".join(
                text.rjust(width) for text, width in zip(row, widths, strict=True)
            )
        )
    return "\n".join(lines)


def _benchmark_summary(directory: Path) -> str:
    timings = directory / "timings.jsonl"
    if not timings.exists():
        return f"no benchmark timings in {directory}"
    records = [json.loads(line) for line in timings.read_text().splitlines() if line]
    rows = []
    for record in records:
        difference = "-"
        if record["mode"].startswith("neighbor"):
            compilation = record["mode"].partition("_")[2]
            dense_path = directory / f"features_dense_{compilation}_{record['set']}.npy"
            neighbor_path = directory / f"features_{record['mode']}_{record['set']}.npy"
            if dense_path.exists() and neighbor_path.exists():
                dense = np.load(dense_path)
                neighbor = np.load(neighbor_path)
                absolute = float(np.abs(neighbor - dense).max())
                relative = absolute / float(np.abs(dense).max())
                difference = f"{absolute:.2e} ({relative:.1e} rel)"
        rows.append(
            [
                record["set"],
                record["mode"],
                f"{record['median_ms']:.1f}",
                f"{record['peak_memory_gib']:.2f}",
                f"{np.mean(record['neighbor_pairs']) / 1e3:.0f}k",
                f"{np.mean(record['dense_pairs']) / 1e3:.0f}k",
                f"{record['collate_ms_cpu']:.1f}",
                f"{record['first_call_seconds']:.1f}",
                str(record["unique_graphs_after_timing"]),
                difference,
            ]
        )
    gpus = sorted({record["gpu"] for record in records})
    header = [
        "set",
        "mode",
        "median ms",
        "peak GiB",
        "nbr pairs",
        "dense pairs",
        "collate ms (CPU)",
        "first call s",
        "graphs",
        "max |neighbor - dense|",
    ]
    return f"GPU: {', '.join(gpus)}\n" + _table(header, rows)


def _smoke_summary(run_directory: Path) -> str:
    throughput = run_directory / "throughput.jsonl"
    if not throughput.exists():
        return f"no throughput.jsonl in {run_directory}"
    records = [json.loads(line) for line in throughput.read_text().splitlines() if line]
    rows = []
    for index, record in enumerate(records):
        has_baseline = index < len(DENSE_BASELINE_STEP_MS)
        rows.append(
            [
                str(record["global_step"]),
                f"{record['step_ms']:.0f}",
                f"{record.get('featurizer_ms', float('nan')):.0f}",
                f"{record.get('featurizer_fraction', float('nan')):.2f}",
                f"{record.get('preprocessor_ms', float('nan')):.0f}",
                str(DENSE_BASELINE_STEP_MS[index]) if has_baseline else "-",
                str(DENSE_BASELINE_FEATURIZER_MS[index]) if has_baseline else "-",
            ]
        )
    header = [
        "step",
        "step ms",
        "featurizer ms",
        "featurizer share",
        "preprocessor ms",
        "dense step ms (H200)",
        "dense featurizer ms (H200)",
    ]
    return _table(header, rows)


def summary(arguments: argparse.Namespace) -> None:
    if arguments.benchmark_dir is not None:
        print("Featurizer benchmark (MACE-POLAR, float32 + cueq, per ~5250-atom batch)")
        print(_benchmark_summary(arguments.benchmark_dir))
        print()
    if arguments.run_directory is not None:
        print(f"Pretraining smoke run (compile off): {arguments.run_directory}")
        print(_smoke_summary(arguments.run_directory))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subcommands.add_parser("benchmark")
    benchmark_parser.add_argument("--mode", choices=MODES, required=True)
    benchmark_parser.add_argument("--model_path", type=Path, required=True)
    benchmark_parser.add_argument("--pcqm4mv2", type=Path, required=True)
    benchmark_parser.add_argument("--geom_drugs", type=Path, required=True)
    benchmark_parser.add_argument("--output_dir", type=Path, required=True)
    benchmark_parser.add_argument("--repeats", type=int, default=20)
    benchmark_parser.set_defaults(run=benchmark)

    smoke_parser = subcommands.add_parser("smoke-config")
    smoke_parser.add_argument("--model_path", type=Path, required=True)
    smoke_parser.add_argument("--pcqm4mv2", type=Path, required=True)
    smoke_parser.add_argument("--geom_drugs", type=Path, required=True)
    smoke_parser.add_argument("--output_base", type=Path, required=True)
    smoke_parser.add_argument("--config_dir", type=Path, required=True)
    smoke_parser.add_argument("--num_workers", type=int, default=12)
    smoke_parser.set_defaults(run=smoke_config)

    summary_parser = subcommands.add_parser("summary")
    summary_parser.add_argument("--benchmark_dir", type=Path, default=None)
    summary_parser.add_argument("--run_directory", type=Path, default=None)
    summary_parser.set_defaults(run=summary)

    arguments = parser.parse_args()
    arguments.run(arguments)


if __name__ == "__main__":
    main()
