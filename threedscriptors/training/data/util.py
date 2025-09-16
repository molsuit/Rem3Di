import os
import statistics as stats
from time import perf_counter

import numcodecs.blosc as blosc
import torch


def worker_init_fn(_):
    # one Blosc thread per worker prevents CPU oversubscription
    try:
        blosc.set_nthreads(1)
    except Exception:
        pass
    # also consider limiting OpenMP threads for NumPy etc.
    os.environ.setdefault("OMP_NUM_THREADS", "1")




def benchmark_loader(dl, warmup=10, max_batches=100, device="cuda"):
    data_times, h2d_times, batch_sizes = [], [], []
    it = iter(dl)

    # warmup
    for _ in range(warmup):
        t0 = perf_counter()
        batch = next(it)
        data_wait = perf_counter() - t0

        t1 = perf_counter()
        batch = batch.to_(device=device, non_blocking=True)  # your .to_ method
        torch.cuda.synchronize()
        h2d = perf_counter() - t1

    # measure
    for _ in range(max_batches):
        t0 = perf_counter()
        batch = next(it)
        data_wait = perf_counter() - t0

        t1 = perf_counter()
        batch = batch.to_(device=device, non_blocking=True)
        torch.cuda.synchronize()
        h2d = perf_counter() - t1

        data_times.append(data_wait)
        h2d_times.append(h2d)
        try:
            batch_sizes.append(len(batch.padding_mask))  # or whatever equals batch size
        except Exception:
            batch_sizes.append(None)

    def summarise(xs):
        return dict(
            mean=stats.mean(xs),
            p50=stats.median(xs),
            p95=sorted(xs)[int(0.95 * len(xs)) - 1],
        )

    return {
        "batches": len(data_times),
        "data_wait": summarise(data_times),
        "h2d": summarise(h2d_times),
        "items_per_s": (
            None if batch_sizes[0] is None else sum(batch_sizes) / sum(data_times)
        ),
    }
