from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from torch.utils.data import DataLoader, Subset

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.data_handling.dataset_creation.dataset_modification import (
    DatasetReconfigurator,
)
from threedscriptors.data_handling.sample import Sample, pretraining_padded_collate_fn
from threedscriptors.training.data import worker_init_fn
from threedscriptors.training.data.samplers import (
    BucketByLengthBatchSampler,
    lengths_from_ptr,
)


@dataclass
class Timing:
    host_batch_times: list[float]
    h2d_times: list[float]
    n_atoms: list[int]


class DataloaderBenchmarkConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    atom_chunk: int = Field(default=8192, ge=1)
    molecule_chunk: int = Field(default=4_096, ge=1)

    bucketed: bool = False
    batch_size: int = Field(default=512, ge=1)
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)

    limit_n_batches: int | None = Field(default=32, ge=1)


class MicrobenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    getitem_mean_s: float
    getitem_atoms_per_s: float
    collate_time_s: float
    collate_batch_size: int

    def summary_lines(self) -> list[str]:
        return [
            f"getitem: mean={self.getitem_mean_s:.6f}s atoms/s={self.getitem_atoms_per_s:.0f}",
            f"collate(batch_size={self.collate_batch_size}): {self.collate_time_s:.6f}s",
        ]


class DataloaderBenchmarkResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: DataloaderBenchmarkConfig
    device: str
    dataset_dir: Path
    batches: int
    atoms_per_batch: float
    atoms_per_second: float
    host_batch_mean_s: float
    host_batch_p50_s: float
    host_batch_p95_s: float
    h2d_mean_s: float | None
    h2d_p50_s: float | None
    h2d_p95_s: float | None
    total_mean_s: float
    total_p50_s: float
    total_p95_s: float
    microbench: MicrobenchmarkResult | None = None

    @field_serializer("dataset_dir")
    def _serialize_dataset_dir(self, dataset_dir: Path) -> str:
        return str(dataset_dir)
    
    def summary(self) -> str:
        if self.batches == 0:
            return "No batches processed."

        lines = [
            f"device={self.device} path={self.dataset_dir}",
            f"batches={self.batches} atoms/batch={self.atoms_per_batch:.0f}",
            (
                f"host_batch: mean={self.host_batch_mean_s:.4f}s "
                f"p50={self.host_batch_p50_s:.4f}s "
                f"p95={self.host_batch_p95_s:.4f}s"
            ),
        ]
        if self.h2d_mean_s is not None:
            lines.append(
                f"h2d:        mean={self.h2d_mean_s:.4f}s "
                f"p50={self.h2d_p50_s:.4f}s "
                f"p95={self.h2d_p95_s:.4f}s"
            )
        lines.append(
            f"total:      mean={self.total_mean_s:.4f}s "
            f"p50={self.total_p50_s:.4f}s "
            f"p95={self.total_p95_s:.4f}s"
        )
        lines.append(f"throughput: {self.atoms_per_second:.0f} atoms/s")
        return "\n".join(lines)

    @classmethod
    def from_timing(
        cls,
        *,
        config: DataloaderBenchmarkConfig,
        device: str,
        dataset_dir: Path,
        timing: Timing,
        microbench: MicrobenchmarkResult | None = None,
    ) -> DataloaderBenchmarkResult:
        if not timing.host_batch_times:
            return cls(
                config=config,
                device=device,
                dataset_dir=dataset_dir,
                batches=0,
                atoms_per_batch=0.0,
                atoms_per_second=0.0,
                host_batch_mean_s=0.0,
                host_batch_p50_s=0.0,
                host_batch_p95_s=0.0,
                h2d_mean_s=None,
                h2d_p50_s=None,
                h2d_p95_s=None,
                total_mean_s=0.0,
                total_p50_s=0.0,
                total_p95_s=0.0,
                microbench=microbench,
            )

        hb = np.asarray(timing.host_batch_times, dtype=np.float64)
        atoms = np.asarray(timing.n_atoms, dtype=np.float64)
        h2d_array = np.asarray(timing.h2d_times, dtype=np.float64)
        h2d = h2d_array if h2d_array.size > 0 else None
        total = hb + (h2d if h2d is not None else 0.0)

        atoms_per_second = atoms.sum() / total.sum() if total.sum() > 0 else 0.0

        h2d_mean = float(h2d.mean()) if h2d is not None else None
        h2d_p50 = float(np.percentile(h2d, 50)) if h2d is not None else None
        h2d_p95 = float(np.percentile(h2d, 95)) if h2d is not None else None

        return cls(
            config=config,
            device=device,
            dataset_dir=dataset_dir,
            batches=len(hb),
            atoms_per_batch=float(atoms.mean()) if atoms.size > 0 else 0.0,
            atoms_per_second=float(atoms_per_second),
            host_batch_mean_s=float(hb.mean()),
            host_batch_p50_s=float(np.percentile(hb, 50)),
            host_batch_p95_s=float(np.percentile(hb, 95)),
            h2d_mean_s=h2d_mean,
            h2d_p50_s=h2d_p50,
            h2d_p95_s=h2d_p95,
            total_mean_s=float(total.mean()),
            total_p50_s=float(np.percentile(total, 50)),
            total_p95_s=float(np.percentile(total, 95)),
            microbench=microbench,
        )


def _count_atoms(sample: Sample) -> int:
    if sample.padding_mask is not None:
        return int((~sample.padding_mask).sum().item())
    if sample.embeddings is not None:
        return int(sample.embeddings.shape[0])
    return 0


class DataloaderBenchmark:
    """
    Create a temporary dataset with updated chunk sizes and benchmark dataloader throughput.
    """

    def __init__(
        self,
        dataset_dir: Path,
        config: DataloaderBenchmarkConfig,
        device: str | None = None,
        *,
        scratch_parent: Path | None = None,
        structures_per_chunk: int = 50_000,
    ):
        self.source_dir = Path(dataset_dir)
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._structures_per_chunk = structures_per_chunk
        self._scratch: TemporaryDirectory | None = None
        self.reconfigured_dir: Path | None = None
        self._dataset: MoleculeDataset | None = None
        self._training_dataset: TrainingMoleculeDataset | None = None
        self._lengths: np.ndarray | None = None
        self._train_indices: np.ndarray | None = None

        self._prepare_dataset(scratch_parent)

    def _prepare_dataset(self, scratch_parent: Path | None) -> None:
        print("Preparing Dataset")
        source_ds = MoleculeDataset.open_existing_dataset_from_dir(self.source_dir)
        src_cfg = source_ds.config
        new_cfg = src_cfg.model_copy(deep=True)
        new_cfg.atom_chunk = self.config.atom_chunk
        new_cfg.molecule_chunk = self.config.molecule_chunk

        if source_ds.smiles is not None:
            source_ds.smiles.close()
        if source_ds.isomeric_smiles is not None:
            source_ds.isomeric_smiles.close()

        scratch_dir = TemporaryDirectory(
            prefix="dataloader_bench_", dir=str(scratch_parent) if scratch_parent else None
        )
        target_dir = Path(scratch_dir.name) / "dataset"
        self._scratch = scratch_dir
        self.reconfigured_dir = target_dir

        try:
            reconfig = DatasetReconfigurator(
                self.source_dir,
                target_dir,
                new_cfg,
                structures_per_chunk=self._structures_per_chunk,
                structure_limit= int(self.config.batch_size * self.config.limit_n_batches* 1.25)
            )
            reconfig.run()
            print("Reconfig completed")
        except Exception:
            self.close()
            raise

        self._dataset = MoleculeDataset.open_existing_dataset_from_dir(target_dir)
        self._training_dataset = TrainingMoleculeDataset(target_dir, pos_emb_getitem)
        self._lengths = lengths_from_ptr(np.asarray(self._dataset.ptr[:]))
        self._train_indices = self._build_train_indices(len(self._lengths))

    @staticmethod
    def _build_train_indices(n_structures: int) -> np.ndarray:
        n_train = int(n_structures * 0.8)
        if n_train <= 0:
            return np.zeros((0,), dtype=np.int64)
        return np.arange(n_train, dtype=np.int64)

    def _build_dataloader(self) -> DataLoader[Sample]:
        if (
            self._training_dataset is None
            or self._train_indices is None
            or self._lengths is None
        ):
            raise RuntimeError("Benchmark dataset not prepared.")

        subset = Subset(self._training_dataset, self._train_indices.tolist())
        kwargs = dict(
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=True,
            num_workers=self.config.num_workers,
            collate_fn=pretraining_padded_collate_fn,
        )

        if self.config.bucketed:
            batch_sampler = BucketByLengthBatchSampler(
                indices=self._train_indices,
                lengths=self._lengths,
                batch_size=self.config.batch_size,
                drop_last=False,
            )
            return DataLoader(subset, batch_sampler=batch_sampler, **kwargs)

        return DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            **kwargs,
        )

    def _collect_timing(self, warmup_batches: int) -> Timing:
        loader = self._build_dataloader()
        timings = Timing(host_batch_times=[], h2d_times=[], n_atoms=[])

        if len(loader) == 0:
            return timings

        iterator = iter(loader)
        try:
            for _ in range(max(0, warmup_batches)):
                next(iterator)
        except StopIteration:
            return timings

        processed = 0
        max_batches = self.config.limit_n_batches or len(loader)
        target_device = torch.device(self.device)

        while processed < max_batches:
            t0 = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            t1 = perf_counter()

            if self.device != "cpu":
                batch.to_(device=target_device, non_blocking=True)
                if target_device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()

            t2 = perf_counter()

            timings.host_batch_times.append(t1 - t0)
            timings.h2d_times.append(max(0.0, t2 - t1))
            timings.n_atoms.append(_count_atoms(batch))
            processed += 1

        return timings

    def run(self, warmup_batches: int = 1) -> DataloaderBenchmarkResult:
        if self.reconfigured_dir is None:
            raise RuntimeError("Benchmark dataset not available.")

        timing = self._collect_timing(warmup_batches)
        return DataloaderBenchmarkResult.from_timing(
            config=self.config,
            device=self.device,
            dataset_dir=self.reconfigured_dir,
            timing=timing,
        )

    def run_microbenchmarks(self, n_iters: int = 1024) -> MicrobenchmarkResult | None:
        if self._training_dataset is None or self._lengths is None:
            raise RuntimeError("Benchmark dataset not prepared.")
        if len(self._lengths) == 0:
            return None

        rng = np.random.default_rng(0)
        idxs = rng.integers(low=0, high=len(self._lengths), size=n_iters, endpoint=False)

        total_atoms = 0
        t0 = perf_counter()
        for i in idxs:
            sample = self._training_dataset[int(i)]
            if sample.embeddings is not None:
                total_atoms += sample.embeddings.shape[0]
        t1 = perf_counter()

        idxs2 = rng.integers(
            low=0, high=len(self._lengths), size=self.config.batch_size, endpoint=False
        )
        samples = [self._training_dataset[int(i)] for i in idxs2]
        t2 = perf_counter()
        _ = pretraining_padded_collate_fn(samples)
        t3 = perf_counter()

        return MicrobenchmarkResult(
            getitem_mean_s=(t1 - t0) / max(1, n_iters),
            getitem_atoms_per_s=total_atoms / max(1e-9, t1 - t0),
            collate_time_s=(t3 - t2),
            collate_batch_size=self.config.batch_size,
        )

    def close(self) -> None:
        if self._dataset is not None:
            if self._dataset.smiles is not None:
                self._dataset.smiles.close()
            if self._dataset.isomeric_smiles is not None:
                self._dataset.isomeric_smiles.close()
        self._dataset = None
        self._training_dataset = None
        self._lengths = None
        self._train_indices = None

        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def __enter__(self) -> DataloaderBenchmark:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        self.close()
