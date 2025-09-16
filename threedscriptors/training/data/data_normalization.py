from dataclasses import dataclass

import torch
from e3nn.o3 import Irreps
from torch import nn
from torch.utils.data import DataLoader

from threedscriptors.configuration.data_config import LabelScalingType
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
)
from threedscriptors.data_handling.sample import Sample, normalization_collate_fn
from threedscriptors.utils.model_utils import (
    get_invariant_indices,
)


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
        dataset : TrainingMoleculeDataset,
        *,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.eps = eps
        self.dataset = dataset  # BaseDataset or IndexedSubset

        #self.stats: NormalizationStats | None = None
        #if getattr(self.dataset, "regression_targets", None) is not None:
        #    self.stats = self._compute_regression_stats()

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


    @torch.no_grad()
    def get_atomic_embedding_normalization_constants(self, irreps : Irreps) -> tuple[torch.Tensor, torch.Tensor]:

        invariant_indices, _ = get_invariant_indices(irreps)

        loader = DataLoader(
                self.dataset,
                batch_size=10000,
                shuffle=False,
                num_workers=16,
                prefetch_factor=12,
                collate_fn= normalization_collate_fn,

            )


        # Accumulators (tiny: shape [1,1,D])
        total_sum = None
        total_sumsq = None
        total_count = 0




        for batch in loader:
            # Cast for stable math; keep batch on whatever device it arrived
            embeddings = batch.embeddings[:,invariant_indices]

            # Reductions over [N]
            emb = embeddings.to(dtype=torch.float64, device="cpu", copy=False)
            s  = emb.sum(dim=0)             # (D_inv,)
            ss = (emb * emb).sum(dim=0)     # (D_inv,)  <-- key fix: sum over dim=0 only
            n  = emb.shape[0]

            if total_sum is None:
                total_sum   = s.clone()
                total_sumsq = ss.clone()
            else:
                total_sum   += s
                total_sumsq += ss

            total_count += n

        mean = total_sum / total_count                                 # (D_inv,)
        var  = (total_sumsq / total_count) - mean.pow(2)             # (D_inv,)
        std  = var.clamp_min(0).sqrt().clamp_min(1e-9)               # (D_inv,)





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

        print(scaling)

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
        for t, m, s in zip(tasks, stats.mean.squeeze(0).tolist(), stats.std.squeeze(0).tolist(), strict=False):
            t.mean = m
            t.std = s


        self.task_configs = tasks

        return stats

    def get_pairwise_differences(self,dataset):

        targets  = torch.log(dataset.regression_targets.reshape(-1,2))
        diffs = targets[:,0] - targets[:,1]

        mean_diff_log = diffs.mean()
        std_diff_logs = diffs.std()

        print(mean_diff_log)
        print(std_diff_logs)


        return mean_diff_log, std_diff_logs
