from threedscriptors.data_handling.dataset import BaseDataset
from threedscriptors.configuration.data_config import DatasetTypes
from threedscriptors.data_handling.data_utils import compute_splits
import numpy as np
from threedscriptors.data_handling.sample import sample_collate_fn, Sample
import torch
from threedscriptors.utils.model_utils import (
    get_mace_calculator_irrep_signature,
    get_invariant_indices,
    split_invariants_equivariants,
)
import math
from typing import Optional, Tuple, Iterable, Union
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import Subset, Dataset
from threedscriptors.data_handling.dataset import BaseDataset
from threedscriptors.data_handling.indexed_subset import IndexedSubset
from torch import nn


from dataclasses import dataclass

from threedscriptors.configuration.data_config import LabelScalingType
from typing import Optional, Tuple


@dataclass
class NormalizationStats:
    mean: torch.Tensor  # [1, T]
    std: torch.Tensor  # [1, T]
    log_mask: torch.Tensor  # [T] bool
    scaling: list[LabelScalingType]

    def to(self, device):
        return NormalizationStats(
            mean=self.mean.to(device),
            std=self.std.to(device),
            log_mask=self.log_mask.to(device),
            scaling=self.scaling,
        )

class DataNormalizationModule(nn.Module):
    """
    Responsibility (only):
      * compute masked mean/std (per task) in the correct space
      * normalize / denormalize regression targets

    Assumptions:
      * regression_masks: 1 == valid, 0 == missing
      * if a task is LOG_Z, any non-positive target has ALREADY been masked out elsewhere
    """

    def __init__(
        self,
        dataset,
        *,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.eps = eps
        self.dataset = dataset  # BaseDataset or IndexedSubset

        self.stats: Optional[NormalizationStats] = None
        if getattr(self.dataset, "regression_targets", None) is not None:
            self.stats = self._compute_regression_stats()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(self, sample: Sample) -> Sample:
        if self.stats is None or sample.regression_targets is None:
            return sample

        rt = sample.regression_targets
        mask = sample.regression_masks.to(rt.dtype)
        sample.regression_targets = self.transform(rt, mask)
        return sample

    def transform(self, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize using precomputed stats; respects mask."""
        assert self.stats is not None

        if mask is None:
            mask = torch.ones_like(y, dtype=y.dtype, device=y.device)

        mean, std = self.stats.mean.to(y.device), self.stats.std.to(y.device)
        log_mask = self.stats.log_mask.to(y.device) # This should probably be moved to the device as a buffer, right?

        if log_mask.any():
            lm = log_mask.view(*(1,) * (y.dim() - 1), -1)
            valid = (mask > 0) & lm
            y = torch.where(valid, torch.log(y.clamp_min(self.eps)), y) # maybe this should be log (x +1 )?? 

        y = (y - mean) / std
        return y * mask

    def inverse_transform(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Inverse normalization (no masking here)."""
        assert self.stats is not None
        mean, std = self.stats.mean.to(y_hat.device), self.stats.std.to(y_hat.device)
        log_mask = self.stats.log_mask.to(y_hat.device)

        y = y_hat * std + mean
        if log_mask.any():
            lm = log_mask.view(*(1,) * (y.dim() - 1), -1)
            y = torch.where(lm, torch.exp(y), y)
        return y

    # ---------------- invariants (input irreps) ------------------------ #

    def get_atomic_embedding_normalization_constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self.dataset.padding_mask  # True = padding, False = real
        embeddings = self.dataset.embeddings

        input_irreps = get_mace_calculator_irrep_signature(
            self.dataset.dataset_config.embedding_model_config.mace_calc
        )

        invariant_indices, _ = get_invariant_indices(input_irreps)
        invariant_embeddings, _ = split_invariants_equivariants(
            embeddings, invariant_indices
        )

        valid_mask = (~padding_mask).unsqueeze(-1)  # True == real atom
        print("Before invariant norms")

        return self.calculate_invariant_normalization_constants_streaming(invariant_embeddings, valid_mask)

    @staticmethod
    def calculate_invariant_normalization_constants(
        invariant_embeddings: torch.Tensor,
        masks: torch.Tensor,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, m = invariant_embeddings, masks

        count = m.sum(dim=(0, 1), keepdim=True).clamp(min=1)
        mean_per_dim = (x * m).sum(dim=(0, 1), keepdim=True) / count
        var = ((x - mean_per_dim) ** 2 * m).sum(dim=(0, 1), keepdim=True) / count
        std_per_dim = torch.sqrt(var).clamp_min(eps)
        return mean_per_dim, std_per_dim
    

    @staticmethod
    @torch.no_grad()
    def calculate_invariant_normalization_constants_streaming(invariant_embeddings, valid_masks,
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = True,
        accumulate_on_cpu: bool = True,
        accum_dtype: torch.dtype = torch.float64,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-dimension mean/std for invariant embeddings with a mask,
        streaming over a DataLoader to avoid OOM.

        Parameters
        ----------
        data
            Either:
            • a DataLoader that yields (x, m) with shapes [B, N, D] and [B, N, 1 or D], or
            • a tuple (embeddings, masks) as tensors (or NumPy arrays) with shapes [B, N, D] and [B, N, 1 or D].
        batch_size
            Used only if `data` is not already a DataLoader.
        num_workers, pin_memory
            DataLoader settings when we construct it for you.
        accumulate_on_cpu
            Keep the small running sums on CPU to minimize GPU memory.
        accum_dtype
            dtype for accumulators; float64 is safest.
        eps
            Lower bound for std to avoid divide-by-zero.

        Returns
        -------
        (mean, std): each shaped [1, 1, D], float32
        """

        # Helper: turn (tensors or numpy arrays) into a DataLoader
        def _as_loader(emb, msk) -> DataLoader:

            # Convert NumPy -> torch if needed
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            if not isinstance(msk, torch.Tensor):
                msk = torch.as_tensor(msk)

            # We don't move to GPU here; we stream batches and move as needed.
            ds = TensorDataset(emb, msk)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )

            return loader

        loader = _as_loader(invariant_embeddings, valid_masks)

        # Accumulators (tiny: shape [1,1,D])
        total_sum = None
        total_sumsq = None
        total_count = None

        def _accum_to_device(t: torch.Tensor) -> torch.Tensor:
            t = t.to(dtype=accum_dtype)
            return t.cpu() if accumulate_on_cpu else t

        for xb, mb in loader:
            # Cast for stable math; keep batch on whatever device it arrived
            xb = xb.float()
            mb = mb.float()
            # Reductions over [B, N]
            s  = (xb * mb).sum(dim=(0, 1), keepdim=True)
            ss = ((xb * xb) * mb).sum(dim=(0, 1), keepdim=True)
            c  = mb.sum(dim=(0, 1), keepdim=True).clamp(min=1)

            s, ss, c = map(_accum_to_device, (s, ss, c))

            if total_sum is None:
                total_sum, total_sumsq, total_count = s, ss, c
            else:
                total_sum   += s
                total_sumsq += ss
                total_count += c

        mean = total_sum / total_count
        var  = total_sumsq / total_count - mean.pow(2)
        std  = var.clamp_min(0).sqrt().clamp_min(eps)


        print(f"mean shape {mean.shape}, std_ shape {std.shape}")
        return mean.to(torch.float32), std.to(torch.float32)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _compute_regression_stats(self) -> NormalizationStats:
        """
        Mask semantics: 1 == valid label, 0 == missing.
        No mask mutation here; we trust upstream sanitation.
        """
        x   = torch.as_tensor(self.dataset.regression_targets, dtype=torch.float32)  # [N, T]
        msk = torch.as_tensor(self.dataset.regression_masks,   dtype=torch.float32)  # [N, T]
        valid = msk > 0

        tasks   = self.dataset.dataset_config.tasks
        scaling = [t.scaling or LabelScalingType.Z for t in tasks]
        log_mask = torch.tensor(
            [s == LabelScalingType.LOG_Z for s in scaling],
            dtype=torch.bool,
            device=x.device,
        )

        if log_mask.any():
            lm = log_mask.unsqueeze(0)
            bad = (valid & lm & (x <= 0)).any()
            if bad:
                raise ValueError(
                    "Found non-positive, unmasked targets for LOG_Z tasks. "
                    "Sanitize them upstream before instantiating DataNormalizationModule."
                )

        # compute stats in the correct space
        y = x.clone()
        if log_mask.any():
            lm = log_mask.unsqueeze(0)
            log_valid = valid & lm
            y = torch.where(log_valid, torch.log(y.clamp_min(self.eps)), y)

        valid_f = valid.to(y.dtype)
        count = valid_f.sum(dim=0).clamp(min=1)
        mean_tasks = (y * valid_f).sum(dim=0) / count
        var_tasks  = ((y - mean_tasks) ** 2 * valid_f).sum(dim=0) / count
        std_tasks  = torch.sqrt(var_tasks).clamp(min=self.eps)

        stats = NormalizationStats(
            mean=mean_tasks.unsqueeze(0),
            std=std_tasks.unsqueeze(0),
            log_mask=log_mask.cpu(),
            scaling=scaling,
        )

        # optionally persist to TaskConfig
        for t, m, s in zip(tasks, stats.mean.squeeze(0).tolist(), stats.std.squeeze(0).tolist()):
            t.mean = m
            t.std = s


        self.task_configs = tasks

        return stats
