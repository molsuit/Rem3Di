from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
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

    def summary(self) -> str:
        if not self.host_batch_times:
            return "No batches processed."
        hb = np.asarray(self.host_batch_times)
        h2d = np.asarray(self.h2d_times) if self.h2d_times else np.zeros_like(hb)
        atoms = np.asarray(self.n_atoms)
        total = hb + h2d

        def fmt(xs: np.ndarray) -> str:
            return (
                f"mean={xs.mean():.4f}s p50={np.percentile(xs,50):.4f}s p95={np.percentile(xs,95):.4f}s"
            )

        lines = []
        lines.append(f"batches={len(hb)}")
        lines.append(f"host_batch: {fmt(hb)}")
        if self.h2d_times:
            lines.append(f"h2d:        {fmt(h2d)}")
        lines.append(f"total:      {fmt(total)}")
        atoms_s = atoms.sum() / total.sum() if total.sum() > 0 else 0.0
        lines.append(f"throughput: {atoms_s:.0f} atoms/s  ({atoms.mean():.0f} atoms/batch avg)")
        return "\n".join(lines)


def count_atoms(sample: Sample) -> int:
    # padding_mask marks valid atoms
    if sample.padding_mask is not None:
        return int(sample.padding_mask.sum().item())
    # fallback
    return int(sample.embeddings.shape[0]) if sample.embeddings is not None else 0


def run_end_to_end(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    limit_batches: int | None,
    bucketed: bool,
    device: str,
) -> Timing:
    root = Path(dataset_dir)
    full = MoleculeDataset.open_existing_dataset_from_dir(root)

    # use fast getitem that reads one contiguous slice per structure
    ds = TrainingMoleculeDataset(root, pos_emb_getitem)

    # Split indices (simple holdout split by molecule ids, consistent with new_dataloaders)
    # For profiling, we only care about training subset
    mol_ptr = np.asarray(full.ptr[:])
    lengths = lengths_from_ptr(mol_ptr)
    n_struct = int(lengths.shape[0])
    # Simple 80/20 split
    n_train = int(n_struct * 0.8)
    train_idx = np.arange(n_train, dtype=np.int64)

    subset = Subset(ds, train_idx.tolist())

    kwargs = dict(
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=pretraining_padded_collate_fn,
    )

    if bucketed:
        # Compute lengths for the subset
        subset_lengths = lengths[train_idx]
        batch_sampler = BucketByLengthBatchSampler(
            indices=train_idx, lengths=lengths, batch_size=batch_size, drop_last=False
        )
        loader = DataLoader(subset, batch_sampler=batch_sampler, **kwargs)
    else:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, **kwargs)

    it = iter(loader)
    device_str = device
    timings = Timing(host_batch_times=[], h2d_times=[], n_atoms=[])
    n_done = 0

    # Warmup one batch to trigger first-open costs
    try:
        _ = next(it)
    except StopIteration:
        return timings

    # Main timed loop
    for _ in range(limit_batches or len(loader)):
        t0 = perf_counter()
        try:
            batch: Sample = next(it)
        except StopIteration:
            break
        t1 = perf_counter()

        # measure host->device transfer separately (optional)
        if device_str != "cpu":
            batch.to_(device=torch.device(device_str), non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        t2 = perf_counter()

        timings.host_batch_times.append(t1 - t0)
        timings.h2d_times.append(max(0.0, t2 - t1))
        timings.n_atoms.append(count_atoms(batch))
        n_done += 1

        if limit_batches is not None and n_done >= limit_batches:
            break

    return timings


def run_component_microbenches(dataset_dir: Path, n_iters: int = 1024, batch_size: int = 512) -> None:
    """Quick micro-benchmarks for dataset getitem and collate cost without workers."""
    root = Path(dataset_dir)
    full = MoleculeDataset.open_existing_dataset_from_dir(root)
    ptr = np.asarray(full.ptr[:])
    lengths = lengths_from_ptr(ptr)
    n_struct = int(lengths.shape[0])

    ds = TrainingMoleculeDataset(root, pos_emb_getitem)

    # Dataset __getitem__ timing (single process)
    rng = np.random.default_rng(0)
    idxs = rng.integers(low=0, high=n_struct, size=n_iters, endpoint=False)
    t0 = perf_counter()
    total_atoms = 0
    for i in idxs:
        s = ds[int(i)]
        total_atoms += s.embeddings.shape[0]
    t1 = perf_counter()
    print(f"getitem: mean={(t1-t0)/n_iters:.6f}s  atoms/s={total_atoms/(t1-t0):.0f}")

    # Collate timing on random list of samples
    idxs2 = rng.integers(low=0, high=n_struct, size=batch_size, endpoint=False)
    samples = [ds[int(i)] for i in idxs2]
    t2 = perf_counter()
    _ = pretraining_padded_collate_fn(samples)
    t3 = perf_counter()
    print(f"collate({batch_size}): {(t3-t2):.6f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile dataloader throughput and timing")
    p.add_argument("--path", type=Path, required=True, help="Zarr dataset root directory")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--limit-batches", type=int, default=32)
    p.add_argument("--bucketed", action="store_true", help="Enable bucketed batching by length")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--micro", action="store_true", help="Run component micro-benchmarks as well")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t = run_end_to_end(
        dataset_dir=args.path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        limit_batches=args.limit_batches,
        bucketed=args.bucketed,
        device=args.device,
    )
    print("\nEnd-to-end timing summary:")
    print(t.summary())

    if args.micro:
        print("\nMicro-benchmarks:")
        run_component_microbenches(args.path, n_iters=1024, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

