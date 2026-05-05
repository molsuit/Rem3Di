"""Run the dataloader benchmark from a yaml config.

Usage:
    uv run scripts/profiling/profile_dataloader.py --config <path-to-yaml>

The yaml is parsed into ProfileBenchmarkConfig (see fields below). The benchmark
copies the source dataset into a temp directory with the requested chunking,
runs the configured DataLoader, and writes a result yaml under output_dir.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pydantic_yaml as pyaml
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from threedscriptors.data_handling.dataset_creation.benchmark_dataloader import (
    DataloaderBenchmark,
    DataloaderBenchmarkConfig,
)


class ProfileBenchmarkConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataset_path: Path
    benchmark: DataloaderBenchmarkConfig
    output_dir: Path = Field(
        default=Path("dataloading_benchmark/evaluations"),
        description="Where to write the result yaml. Relative paths resolve "
        "against the current working directory.",
    )
    device: str | None = None
    run_micro: bool = False
    micro_iters: int = 1024
    warmup_batches: int = 10
    structures_per_chunk: int = 50_000
    scratch_parent: Path | None = None

    def resolved_device(self) -> str:
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DataLoader throughput benchmark."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the benchmark yaml."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Optional override for the dataset path (e.g. compute-node-staged data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_cfg = pyaml.parse_yaml_file_as(ProfileBenchmarkConfig, args.config)

    dataset_path = (
        args.dataset_path if args.dataset_path is not None else script_cfg.dataset_path
    )

    output_dir = script_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with DataloaderBenchmark(
        dataset_dir=dataset_path,
        config=script_cfg.benchmark,
        device=script_cfg.resolved_device(),
        scratch_parent=script_cfg.scratch_parent,
        structures_per_chunk=script_cfg.structures_per_chunk,
    ) as benchmark:
        print("Starting benchmark")
        result = benchmark.run(warmup_batches=script_cfg.warmup_batches)
        print("\nEnd-to-end timing summary:")
        print(result.summary())

        if script_cfg.run_micro:
            print("\nMicro-benchmarks:")
            micro = benchmark.run_microbenchmarks(script_cfg.micro_iters)
            if micro is None:
                print("No data for micro-benchmarks.")
            else:
                for line in micro.summary_lines():
                    print(line)
                result = result.model_copy(update={"microbench": micro})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"benchmark_{timestamp}.yaml"
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(result.model_dump(mode="json"), fh, sort_keys=False)
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
