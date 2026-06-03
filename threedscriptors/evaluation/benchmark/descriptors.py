"""Descriptor calculators for the descriptor-probe benchmark.

A *descriptor calculator* turns a prepared :class:`MoleculeDataset` into an
``(N, D)`` matrix aligned with the dataset's structure order. SMILES-based
calculators read ``dataset.get_smiles_per_structure()``; structure-based
calculators use a PyTorch :class:`~torch.utils.data.DataLoader` over
:class:`TrainingMoleculeDataset` to batch through positions / atomic_numbers
from the zarr — that's the "specific dataloader" path the user called out.

Adding a new structure-based model = (a) subclass :class:`DescriptorCalculator`
implementing ``calculate``, (b) add a pydantic config variant to the
``DescriptorConfig`` union with a ``.build()`` factory. Both are local.

The optional ``train_indices`` kwarg on ``calculate`` is for future fitted
descriptors (e.g. learned PCA) that must only fit on the train split.
:class:`EcfpCalculator` and :class:`RemediCalculator` ignore it because they
are deterministic / pre-trained.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from ase import Atoms
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Subset

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from threedscriptors.data_handling.sample import Sample, yield_molecules_collate_fn


class DescriptorCalculator(ABC):
    """One method, one matrix.

    Concrete subclasses are responsible for *how* they read the dataset
    (SMILES vs DataLoader over the zarr) but must produce a result aligned
    with the dataset's structure order so the runner can slice it by the
    materialized ``split`` column.
    """

    @abstractmethod
    def calculate(
        self,
        dataset: MoleculeDataset,
        *,
        train_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return shape ``(dataset.N_structures, descriptor_dim)``."""


# -- EcfpCalculator: SMILES-based via molfeat --------------------------------


class EcfpCalculator(DescriptorCalculator):
    def __init__(self, *, fingerprint: str = "ecfp", length: int = 2048) -> None:
        self.fingerprint = fingerprint
        self.length = length

    def calculate(
        self,
        dataset: MoleculeDataset,
        *,
        train_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        del train_indices  # ECFP is deterministic — no fit step
        from molfeat.trans.fp import FPVecTransformer

        featurizer = FPVecTransformer(kind=self.fingerprint, length=self.length)
        X = featurizer(dataset.get_smiles_per_structure())
        return np.asarray(X, dtype=np.float32)


# -- RemediCalculator: 3D-structure via REM3DI encoder -----------------------
# Uses TrainingMoleculeDataset + DataLoader over the zarr's positions /
# atomic_numbers — this is the "specific dataloader" pattern that any future
# structure-based model should mirror.


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
        from threedscriptors.configuration.architecture_config import (
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
        same encoder so the resulting vector lives in the same space as a built
        :class:`~threedscriptors.evaluation.retrieval.vector_store.VectorStore`.

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
        samples = [
            self._atoms_to_sample(a, total_charge, multiplicity) for a in atoms
        ]
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


# -- Pydantic configs --------------------------------------------------------


class EcfpConfig(BaseModel):
    descriptor_kind: Literal["ecfp"] = "ecfp"
    # Human-readable identifier used in the cache filename. Change this if you
    # change `fingerprint` or `length` so different ECFP variants don't clash.
    name: str = "ecfp"
    fingerprint: str = "ecfp"
    length: int = 2048

    def build(self) -> EcfpCalculator:
        return EcfpCalculator(fingerprint=self.fingerprint, length=self.length)


class RemediConfig(BaseModel):
    descriptor_kind: Literal["remedi"] = "remedi"
    # User-supplied identifier used in the cache filename — usually the model
    # checkpoint name (e.g. "remedi-geom-350k").
    name: str
    model_dir: Path
    batch_size: int = 64
    device: Literal["cuda", "cpu"] = "cuda"
    # Optional override for the MACE foundation-model path baked into the
    # checkpoint config (which is absolute and machine-specific). Set this when
    # running a checkpoint on a different machine than it was trained on.
    mace_model_path: Path | None = None

    def build(self) -> RemediCalculator:
        return RemediCalculator(
            model_dir=self.model_dir,
            batch_size=self.batch_size,
            device=self.device,
            mace_model_path=self.mace_model_path,
        )


DescriptorConfig = Annotated[
    EcfpConfig | RemediConfig, Field(discriminator="descriptor_kind")
]


# -- Cache helper ------------------------------------------------------------


def compute_and_cache(
    config: EcfpConfig | RemediConfig,
    dataset: MoleculeDataset,
    cache_dir: Path,
    dataset_id: str,
) -> np.ndarray:
    """Compute the descriptor matrix, caching by ``(dataset_id, config.name)``.

    The cache key uses the user-supplied ``config.name``; the user changes it
    when descriptor params change (or deletes the npz to force recompute).
    Cache files live at ``{cache_dir}/{dataset_id}__{config.name}.npz``.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{dataset_id}__{config.name}.npz"
    if cache_path.exists():
        return np.asarray(np.load(cache_path)["X"])
    X = config.build().calculate(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X)
    return X
