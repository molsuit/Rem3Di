from typing import Union

import torch
import torch.nn as nn

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.data_handling.sample import PreprocessedSample
from threedscriptors.model.preprocessing.chiral_embedding_model import (
    ChiralEmbeddingModel,
)
from threedscriptors.utils.model_utils import (
    get_equivariant_irreps,
    get_invariant_indices,
    split_invariants_equivariants,
)


class RMSLayerNorm(nn.Module):
    def __init__(self, num_blocks: int, eps: float = 1e-6):
        """
        RMS-style layer norm for equivariant (type-L) blocks.

        Args:
          num_channels: number of irreducible blocks C
          eps: small constant to avoid div/0

            Only supports l=1 irreps for now

        """
        super().__init__()
        # one learnable scale per channel/block
        self.gamma = nn.Parameter(torch.ones(num_blocks, 1))
        self.eps = eps

    def forward(self, S_e: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: tensor of shape (..., C, D)

        Returns:
          normalized tensor of the same shape
        """
        # 1) compute per-block L2 norm over the last dim:
        #    shape: (..., C, 1)

        B, N, D = S_e.shape
        C = D // 3

        S_e = S_e.view(B, N, C, 3)

        block_norms = torch.linalg.norm(S_e, dim=-1, keepdim=True)

        # 2) compute RMS of those norms *across* the C channels:
        #    shape: (..., 1, 1)
        rms = torch.sqrt(
            torch.mean(block_norms.pow(2), dim=-2, keepdim=True) + self.eps
        )

        if padding_mask is not None:
            rms = rms.masked_fill(padding_mask[..., None, None], 1.0)

        # 3) divide each block by the shared rms and apply per-channel scale
        #    broadcasting gamma over any leading dims and over D
        out = (S_e / rms) * self.gamma

        if padding_mask is not None:
            out = out.masked_fill(padding_mask[..., None, None], 0.0)

        return out.reshape(B, N, D)



class PrecomputedInvariantNormalization(nn.Module):
    def __init__(self, invariant_dimension: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Create the buffers ONCE and never replace them.
        self.register_buffer("mean", torch.zeros(1, 1, invariant_dimension), persistent=True)
        self.register_buffer("std",  torch.ones(1, 1, invariant_dimension), persistent=True)

        # Track whether stats were set; not persisted (purely runtime convenience).
        self.register_buffer("_stats_set", torch.tensor(False), persistent=False)

    @torch.no_grad()
    def set_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        overwrite: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Safely register normalization statistics without replacing buffers.

        Parameters
        ----------
        mean, std : tensors/arrays broadcastable to (1, 1, dim)
        overwrite : allow changing stats after they were set once
        device, dtype : (optional) explicit casting targets
        """
        if not overwrite and bool(self._stats_set.item()):
            raise RuntimeError(
                "Normalization stats are already set. "
                "Pass overwrite=True to replace them."
            )




        mean = torch.as_tensor(mean, device=device or self.mean.device, dtype=dtype or self.mean.dtype)
        std  = torch.as_tensor(std,  device=device or self.std.device,  dtype=dtype or self.std.dtype)


            # Coerce to (1, 1, D)
        if mean.dim() == 1:
            mean = mean.view(1, 1, -1)
        elif mean.dim() == 2:
            mean = mean.unsqueeze(0)          # (1, 1, D) if it was (1, D)
        # otherwise expect already (1,1,D)
        if std.dim() == 1:
            std = std.view(1, 1, -1)
        elif std.dim() == 2:
            std = std.unsqueeze(0)



        if mean.shape != self.mean.shape:
            raise ValueError(f"mean has shape {mean.shape}, expected {self.mean.shape}")
        if std.shape != self.std.shape:
            raise ValueError(f"std has shape {std.shape}, expected {self.std.shape}")

        # Avoid zero std -> add eps when used (or clamp here).
        self.mean.data.copy_(mean)
        self.std.data.copy_(std)
        self._stats_set.fill_(True)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        # x: (B, N, dim), padding_mask: (B, N) with True = pad
        x = (x - self.mean) / (self.std + self.eps)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask[..., None], 0.0)


        return x


class OnTheFlyInvariantNormalization(nn.Module):
    def __init__(
        self,
        invariant_dimension: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        warmup_batches: int = 1000,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.warmup_batches = warmup_batches

        self.register_buffer("running_mean", torch.zeros(invariant_dimension))
        self.register_buffer("running_var", torch.ones(invariant_dimension))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen", torch.tensor(False))

    def _update_running_stats(self, x: torch.Tensor, padding_mask: torch.Tensor | None):
        with torch.no_grad():
        # x: (B, N, D), padding_mask: (B, N), True = padded
            if padding_mask is not None:
                valid = ~padding_mask.bool()
                if not valid.any():
                    return
                x_valid = x[valid]  # (num_valid, D)
            else:
                x_valid = x.reshape(-1, x.size(-1))

            if x_valid.numel() == 0:
                return

            batch_mean = x_valid.mean(dim=0)
            batch_var = x_valid.var(dim=0, unbiased=False)

            if self.num_batches_tracked == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.lerp_(batch_mean, self.momentum)
                self.running_var.lerp_(batch_var, self.momentum)

            self.num_batches_tracked += 1
            if self.num_batches_tracked >= self.warmup_batches:
                self.frozen.fill_(True)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, D)
        if self.training and not bool(self.frozen):
            self._update_running_stats(x, padding_mask)

        mean = self.running_mean.view(1, 1, -1)
        std = (self.running_var + self.eps).sqrt().view(1, 1, -1)

        x_norm = (x - mean) / std

        if padding_mask is not None:
            # True = padded → zero out normalized values there
            mask = ~padding_mask.bool()              # (B, N)
            x_norm = x_norm * mask.unsqueeze(-1)     # (B, N, D)

        return x_norm

InvariantNormalization = OnTheFlyInvariantNormalization |  PrecomputedInvariantNormalization


class AtomicDescriptorPreprocessor(nn.Module):
    """
    Pretreats the calculated embeddings with two possible strategies. 1. Get only the invariant part. 2. Add the pseudoscalar.
    """

    def __init__(self, preprocess_config: EmbeddingPreprocessConfig, invariant_normalization : InvariantNormalization):
        super().__init__()

        self.config = preprocess_config
        self.invariant_indices, self.invariant_irreps = get_invariant_indices(
            self.config.input_irreps
        )

        self.equivariant_irreps = get_equivariant_irreps(self.config.input_irreps)

        assert all([l == 1 for l in self.equivariant_irreps.ls])

        self.invariant_normalization = invariant_normalization

        self.equivariant_rms_norm = RMSLayerNorm(num_blocks = self.equivariant_irreps.num_irreps)

        self.chiral_embedding_model = ChiralEmbeddingModel        (invariant_irreps= self.invariant_irreps, equivariant_irreps=self.equivariant_irreps, pseudoscalar_dimension=self.config.pseudoscalar_dimension, chiral_embedding_dim=self.config.chiral_embedding_dimension, gated = self.config.gated, dtype=torch.float32)

        self.has_chiral_embedding = self.config.pseudoscalars


    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) -> PreprocessedSample:

        invariants, equivariants = split_invariants_equivariants(embeddings, self.invariant_indices)

        normalized_invariants = self.invariant_normalization(invariants, padding_mask)


        if self.has_chiral_embedding:
            normalized_equivariants = self.equivariant_rms_norm(equivariants, padding_mask)

            chiral_embedding = self.chiral_embedding_model(normalized_invariants, normalized_equivariants, padding_mask)


            normalized_invariants = normalized_invariants.to(dtype = torch.float32)
            return PreprocessedSample(preprocessed_atomic_embeddings=torch.cat((normalized_invariants, chiral_embedding), dim = -1), chiral_embeddings=chiral_embedding)

        else:
            normalized_invariants = normalized_invariants.to(dtype = torch.float32)

            return PreprocessedSample(preprocessed_atomic_embeddings=normalized_invariants)
