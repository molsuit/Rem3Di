from dataclasses import dataclass

import torch


@dataclass
class MolecularDescriptor:
    """A batched per-molecule descriptor as a sequence of summary tokens.

    Aggregators that emit a single summary (MeanPool, AttnPool) wrap their
    output as a length-1 sequence. PMA emits one token per seed. Consumers
    that want a flat fixed-size vector use `.flat`; consumers that want to
    cross-attend to the tokens (e.g. the decoder) use `.tokens`.

    Attributes
    ----------
    tokens : torch.Tensor of shape (B, L, D)
    """

    tokens: torch.Tensor

    def __post_init__(self) -> None:
        if self.tokens.dim() != 3:
            raise ValueError(
                f"MolecularDescriptor.tokens must be (B, L, D); got shape "
                f"{tuple(self.tokens.shape)}"
            )

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.tokens.shape[1]

    @property
    def dim(self) -> int:
        return self.tokens.shape[2]

    @property
    def flat_dim(self) -> int:
        return self.seq_len * self.dim

    @property
    def flat(self) -> torch.Tensor:
        """Flatten the seed sequence to (B, L*D)."""
        return self.tokens.reshape(self.batch_size, self.flat_dim)

    def to(self, device: torch.device | str) -> "MolecularDescriptor":
        return MolecularDescriptor(tokens=self.tokens.to(device))

    def detach(self) -> "MolecularDescriptor":
        return MolecularDescriptor(tokens=self.tokens.detach())

    def cpu(self) -> "MolecularDescriptor":
        return self.to("cpu")

    def __getitem__(self, idx) -> "MolecularDescriptor":
        sliced = self.tokens[idx]
        if sliced.dim() == 2:
            sliced = sliced.unsqueeze(0)
        return MolecularDescriptor(tokens=sliced)
