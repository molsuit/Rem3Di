"""The Rem3Di descriptor calculator: a trained encoder over a dataset's structures.

:class:`RemediCalculator` streams a :class:`MoleculeDataset`'s positions and
atomic numbers through :class:`TrainingMoleculeDataset` and a PyTorch
:class:`DataLoader` into an ``(N, D)`` matrix aligned with the dataset's
structure order. :class:`RemediConfig` is its yaml form and carries the
checkpoint identity. ``remedi_eval`` never imports this module; Rem3Di enters
the evaluation as precomputed embeddings (``BENCHMARK_DATA_FORMAT.md`` §7.5).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ase import Atoms
from pydantic import BaseModel
from torch.utils.data import DataLoader, Subset

from remedi.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from remedi.data_handling.sample import Sample, yield_molecules_collate_fn
from remedi_data.bundle import sha256_of_file
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.base import DescriptorCalculator


class RemediCalculator(DescriptorCalculator):
    def __init__(
        self,
        *,
        model_dir: Path,
        batch_size: int = 64,
        device: Literal["cuda", "cpu"] = "cuda",
        mace_model_path: Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.device_str = device
        # Override for the MACE foundation-model path baked into the checkpoint
        # config. The training-time path is absolute and machine-specific, so a
        # checkpoint moved to another cluster needs the local path supplied here.
        self.mace_model_path = (
            Path(mace_model_path) if mace_model_path is not None else None
        )
        self._model = None  # lazy: load on first calculate() to keep import cheap

    def _ensure_model_loaded(self):
        if self._model is not None:
            return
        from remedi.configuration.architecture_config import (
            EncoderOnlyArchitectureConfig,
        )

        cfg = EncoderOnlyArchitectureConfig.from_encoder_yaml(self.model_dir)
        if self.mace_model_path is not None:
            if cfg.mace_config is None:
                raise ValueError(
                    "mace_model_path override given but the checkpoint config "
                    "has no mace_config to apply it to."
                )
            cfg.mace_config.model_path = self.mace_model_path
        model = cfg.build()
        model.encoder.load_state_dict(torch.load(self.model_dir / "encoder.pth"))
        model.preprocessor.atomic_preprocessor.load_state_dict(
            torch.load(self.model_dir / "atomic_preprocessor.pth")
        )
        model.preprocessor.geometric_preprocessor.load_state_dict(
            torch.load(self.model_dir / "geometric_preprocessor.pth")
        )
        self._model = model.eval()

    def _resolve_device(self) -> torch.device:
        if self.device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "RemediCalculator requested CUDA but torch.cuda.is_available() "
                "is False. Set device='cpu' explicitly to opt into CPU."
            )
        return torch.device(self.device_str)

    def calculate(
        self,
        dataset: MoleculeDataset,
        *,
        train_indices: np.ndarray | None = None,
        limit: int | None = None,
    ) -> np.ndarray:
        """Embed the dataset into ``(N, D)`` (or ``(limit, D)`` for a prefix).

        ``limit`` embeds only the first ``limit`` structures — used for cheap
        smoke runs so a capped store does not pay to embed the whole dataset.
        """
        del train_indices  # REM3DI is pre-trained — no fit step
        self._ensure_model_loaded()
        assert self._model is not None  # narrowed by _ensure_model_loaded
        device = self._resolve_device()
        model = self._model.to(device)

        train_ds: TrainingMoleculeDataset | Subset = (
            TrainingMoleculeDataset.from_molecule_dataset(
                dataset, get_item=atoms_getitem
            )
        )
        if limit is not None and limit < len(train_ds):
            train_ds = Subset(train_ds, range(limit))
        n = len(train_ds)
        loader: Iterable[Sample] = DataLoader(
            train_ds,
            batch_size=min(self.batch_size, max(n, 1)),
            shuffle=False,
            drop_last=False,
            collate_fn=yield_molecules_collate_fn,
        )
        return self._embed_batches(model, device, loader, n)

    def embed_atoms(
        self,
        atoms: list[Atoms],
        *,
        total_charge: float = 0.0,
        multiplicity: float = 1.0,
    ) -> np.ndarray:
        """Embed an explicit list of ASE :class:`Atoms` into ``(len(atoms), D)``.

        This is the ad-hoc counterpart to :meth:`calculate` (which streams a
        whole on-disk dataset): it takes molecules that are *not* in the zarr —
        e.g. a brand-new query molecule conformer — and pushes them through the
        same encoder so the resulting vectors live in the same space as the
        cached matrix :meth:`calculate` produces.

        ``total_charge`` / ``multiplicity`` default to neutral singlet and apply
        to every molecule; supply per-molecule values via the ASE ``info`` dict
        upstream if you need them to vary.
        """
        self._ensure_model_loaded()  # also makes _flat_dim() safe for empty input
        assert self._model is not None
        if not atoms:
            return np.zeros((0, self._flat_dim()), dtype=np.float32)
        device = self._resolve_device()
        model = self._model.to(device)

        n = len(atoms)
        bs = min(self.batch_size, max(n, 1))
        samples = [self._atoms_to_sample(a, total_charge, multiplicity) for a in atoms]
        batches = [
            yield_molecules_collate_fn(samples[i : i + bs]) for i in range(0, n, bs)
        ]
        return self._embed_batches(model, device, batches, n)

    @staticmethod
    def _atoms_to_sample(
        atoms: Atoms, total_charge: float, multiplicity: float
    ) -> Sample:
        pos = torch.from_numpy(np.asarray(atoms.get_positions(), dtype=np.float32))
        num = torch.from_numpy(np.asarray(atoms.get_atomic_numbers()))
        return Sample(
            atomic_positions=pos,
            atomic_numbers=num,
            total_charge=torch.tensor(float(total_charge)),
            multiplicity=torch.tensor(float(multiplicity)),
        )

    def _flat_dim(self) -> int:
        assert self._model is not None
        aggregator = self._model.encoder.aggregator
        # PyTorch's nn.Module attribute access is typed `Tensor|Module` in the
        # stubs even when the runtime value is a plain int (REM3DI's
        # aggregator stores seq_len/d_out as ints) — getattr sidesteps the
        # stub without runtime overhead.
        return int(getattr(aggregator, "seq_len")) * int(  # noqa: B009
            getattr(aggregator, "d_out")  # noqa: B009
        )

    def _embed_batches(
        self,
        model,
        device: torch.device,
        batches: Iterable[Sample],
        n: int,
    ) -> np.ndarray:
        """Stream collated ``Sample`` batches through the encoder into ``(n, D)``."""
        descriptors = torch.zeros((n, self._flat_dim()), dtype=torch.float32)
        with torch.no_grad():
            cursor = 0
            for samples in batches:
                samples.to_(device)
                batch_descriptor = model(samples).molecular_descriptor.flat.cpu()
                bs = batch_descriptor.shape[0]
                descriptors[cursor : cursor + bs] = batch_descriptor
                cursor += bs
        return descriptors.numpy()


class RemediConfig(BaseModel):
    # The descriptor name of the embeddings files, usually the checkpoint name
    # (e.g. "remedi-geom-350k"); the manifest reads them back under this name.
    name: str
    model_dir: Path
    batch_size: int = 64
    device: Literal["cuda", "cpu"] = "cuda"
    # Optional override for the MACE foundation-model path baked into the
    # checkpoint config (which is absolute and machine-specific). Set this when
    # running a checkpoint on a different machine than it was trained on.
    mace_model_path: Path | None = None
    # Filled in on first use by :meth:`identity` and recorded as the
    # descriptor_identity of every embeddings file, so a results table traces
    # back to the weights that produced it. Set it by hand only to pin an
    # identity deliberately.
    checkpoint_sha256: str | None = None

    def identity(self) -> str:
        """sha256 over the checkpoint weights, computed once and memoised.

        The weights are the descriptor: ``batch_size`` and ``device`` change
        nothing about the matrix and must not force a re-embedding, while two
        checkpoints under one ``name`` must never share embeddings.

        Raises:
            FileNotFoundError: if ``model_dir`` holds no ``.pth`` file.
        """
        if self.checkpoint_sha256 is None:
            self.checkpoint_sha256 = self._hash_checkpoint()
        return self.checkpoint_sha256

    def _hash_checkpoint(self) -> str:
        checkpoint_files = sorted(Path(self.model_dir).glob("*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"{self.model_dir} holds no .pth checkpoint file, so the "
                f"descriptor {self.name!r} has no identity to key its cache on"
            )
        digest = hashlib.sha256()
        for path in checkpoint_files:
            digest.update(f"{path.name}:{sha256_of_file(path)}\n".encode())
        return digest.hexdigest()

    def build(self) -> RemediCalculator:
        return RemediCalculator(
            model_dir=self.model_dir,
            batch_size=self.batch_size,
            device=self.device,
            mace_model_path=self.mace_model_path,
        )
