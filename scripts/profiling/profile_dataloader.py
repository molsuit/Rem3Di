from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threedscriptors.data_handling.dataset_creation.benchmark_dataloader import (
    DataloaderBenchmark,
    DataloaderBenchmarkConfig,
)


class ProfileBenchmarkConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_path: Path
    benchmark: DataloaderBenchmarkConfig = Field(
        default_factory=DataloaderBenchmarkConfig
    )
    device: str | None = None
    run_micro: bool = False
    micro_iters: int = 1024
    warmup_batches: int = 10
    structures_per_chunk: int = 50_000
    scratch_parent: Path | None = None

    def resolved_device(self) -> str:
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")


SCRIPT_CONFIG_DATA: dict = {
    "dataset_path": "/local/data/public/snw30/geom_drugs",
    "benchmark": {
        "atom_chunk": 1000,
        "molecule_chunk": 20,
        "bucketed": False,
        "batch_size": 256,
        "num_workers": 16,
        "prefetch_factor": 4,
        "limit_n_batches": 50,
    },
    "run_micro": False,
    "micro_iters": 1024,
    "warmup_batches": 10,
    "structures_per_chunk": 50_000,
    "scratch_parent": "/local/data/public/snw30/dataloading_benchmark",
}


def load_script_config() -> ProfileBenchmarkConfig:
    return ProfileBenchmarkConfig.model_validate(SCRIPT_CONFIG_DATA)


def main() -> int:
    try:
        script_cfg = load_script_config()
    except ValidationError as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc

    with DataloaderBenchmark(
        dataset_dir=script_cfg.dataset_path,
        config=script_cfg.benchmark,
        device=script_cfg.resolved_device(),
        scratch_parent=script_cfg.scratch_parent,
        structures_per_chunk=script_cfg.structures_per_chunk,
    ) as benchmark:
        print("Starting Benchmark")
        result = benchmark.run(warmup_batches=script_cfg.warmup_batches)

        print("\nEnd-to-end timing summary:")
        print(result.summary())
        evaluations_dir = Path(
            "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/dataloading_benchmark/evaluations"
        )
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = evaluations_dir / f"benchmark_{timestamp}.yaml"
        with output_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(result.model_dump(), fh)

        if script_cfg.run_micro:
            print("\nMicro-benchmarks:")
            micro = benchmark.run_microbenchmarks(script_cfg.micro_iters)
            if micro is None:
                print("No data for micro-benchmarks.")
            else:
                for line in micro.summary_lines():
                    print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
