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

from torch.utils.data import Subset
from torch import nn

class DataNormalizationModule():
    def __init__(self, dataset):
        # If this is a Subset, grab the underlying dataset and its indices
        if isinstance(dataset, Subset):
            self.base_ds = dataset.dataset
            self.indices = torch.as_tensor(dataset.indices)
        else:
            self.base_ds = dataset
            self.indices = None

        # compute mean/std on the (possibly indexed) base dataset
        self._compute_mean_std()

    def __call__(self, sample: Sample):
        rt = sample.regression_targets  # [..., T]
        mask = sample.regression_masks.to(rt.dtype)

        # In‐place normalization
        rt.sub_(self.mean_tasks).div_(self.std_tasks)
        rt.mul_(mask)

        return sample

    def _take(self, arr):
        """Index into arr if we have a Subset, else return it directly"""
        if self.indices is None:
            return arr
        return arr[self.indices]

    def _compute_mean_std(self):
        # Pull out the full arrays (or the subset of them)
        x = self._take(self.base_ds.regression_targets)  # shape [N, T]
        mask = self._take(self.base_ds.regression_masks)  # shape [N, T]

        # Ensure tensors
        x = torch.as_tensor(x, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        # Compute masked mean/std along dim=0
        count = mask.sum(dim=0).clamp(min=1)  # [T]
        mean_tasks = (x * mask).sum(dim=0) / count
        var_tasks = ((x - mean_tasks) ** 2 * mask).sum(dim=0) / count
        std_tasks = torch.sqrt(var_tasks).clamp(min=1e-12)

        # Unsqueeze so they broadcast over any leading batch dims
        self.mean_tasks = mean_tasks.unsqueeze(0)  # [1, T]
        self.std_tasks = std_tasks.unsqueeze(0)  # [1, T]

    def normalize_regression_targets(self, mean_targets, std_targets):

        print("Sizes")
        print(self.dataset.embeddings.shape)
        print(self.dataset.regression_masks.shape)
        print(self.dataset.regression_targets.shape)

        assert self.dataset.regression_targets is not None

        dataset_tasks = self.dataset.dataset_config.get_task_names()

        if isinstance(self.dataset.regression_targets, torch.Tensor):
            regression_targets = self.dataset.regression_targets.detach().cpu().numpy()

        if self.dataset.dataset_config.regression_is_normalized:
            print("Dataset was already normalized")
            return

        log_scaling_mask = np.array(
            [
                tc.scaling == LabelScalingType.LOG
                for tc in self.dataset.dataset_config.tasks
            ]
        ).reshape(-1, len(self.dataset.dataset_config.get_task_names()))

        reg_masks = self.dataset.regression_masks.detach().cpu().numpy()

        print(reg_masks.shape)
        log_mask = np.logical_and(log_scaling_mask, reg_masks)

        print(log_mask.shape)

        if not np.all(regression_targets[log_mask] > 0):
            remove_rows, _ = np.where((log_mask) & (regression_targets <= 0))

            n_rows = self.dataset.embeddings.shape[0]
            print(self.dataset.embeddings.shape)
            keep = np.ones(n_rows, dtype=bool)
            keep[remove_rows] = False

            self.dataset.embeddings = self.dataset.embeddings[keep, :, :]
            self.dataset.molecules = [
                m for m, k in zip(self.dataset.molecules, keep) if k
            ]

            reg_masks = reg_masks[keep, :]
            log_mask = log_mask[keep, :]
            regression_targets = regression_targets[keep, :]
            self.dataset.regression_masks = self.dataset.regression_masks[keep, :]
            self.dataset.padding_mask = self.dataset.padding_mask[keep, :]
            self.dataset.atomic_positions = self.dataset.atomic_positions[keep, :, :]

            self.dataset.dataset_config.N_molecules = self.dataset.embeddings.shape[0]

        assert np.all(regression_targets[log_mask] > 0)

        regression_targets[log_mask] = np.log(regression_targets[log_mask])

        if mean_targets is None:
            mean_targets = np.mean(
                regression_targets, axis=0, where=self.dataset.regression_masks
            )

        else:
            assert list(mean_targets.keys()) == dataset_tasks
            mean_targets = np.array(list(mean_targets.values()))

        if std_targets is None:
            std_targets = np.std(
                regression_targets, axis=0, where=self.dataset.regression_masks
            )

        else:
            assert list(std_targets.keys()) == dataset_tasks
            std_targets = np.array(list(std_targets.values()))

        print(f"Mean Targets {mean_targets}")
        print(f"Std Targets {std_targets}")

        self.dataset.regression_targets = (
            regression_targets - mean_targets
        ) / std_targets

        for task, task_mean, task_std in zip(
            self.dataset.dataset_config.tasks,
            mean_targets.tolist(),
            std_targets.tolist(),
            strict=False,
        ):

            task.mean = task_mean
            task.std = task_std

        self.dataset.dataset_config.regression_is_normalized = True

        self.dataset.regression_targets = torch.Tensor(self.dataset.regression_targets)

    def get_atomic_embedding_normalization_constants(self):

        padding_mask = self._take(self.base_ds.padding_mask)
        embeddings = self._take(self.base_ds.embeddings)

        input_irreps = get_mace_calculator_irrep_signature(
            self.base_ds.dataset_config.embedding_model_config.mace_calc
        )

        invariant_indices, _ = get_invariant_indices(input_irreps)
        invariant_embeddings, _ = split_invariants_equivariants(
            embeddings, invariant_indices
        )

        masks = padding_mask.unsqueeze(-1) == 0

        inv_mean_per_dim, inv_std_per_dim = (
            self.calculate_invariant_normalization_constants(
                invariant_embeddings, masks
            )
        )

        return inv_mean_per_dim, inv_std_per_dim

    @staticmethod
    def calculate_invariant_normalization_constants(invariant_embeddings, masks):

        x = invariant_embeddings
        m = masks

        # 1) count of valid elems per‐dim
        count = m.sum(dim=(0, 1), keepdim=True).clamp(min=1)  # [1,1,D]

        # 2) masked sum
        sum_ = (x * m).sum(dim=(0, 1), keepdim=True)  # [1,1,D]

        # 3) mean
        mean_per_dim = sum_ / count  # [1,1,D]

        # 4) variance
        sq_diff = (x - mean_per_dim) ** 2 * m
        var = sq_diff.sum(dim=(0, 1), keepdim=True) / count

        # 5) std
        std_per_dim = torch.sqrt(var)

        return mean_per_dim, std_per_dim
