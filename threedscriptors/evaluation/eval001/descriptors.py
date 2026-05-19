from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def compute_ecfp(smiles: list[str], *, kind: str = "ecfp:4", length: int = 2048) -> np.ndarray:
    from molfeat.trans.fp import FPVecTransformer

    featurizer = FPVecTransformer(kind=kind, length=length)
    X = featurizer(smiles)
    return np.asarray(X, dtype=np.float32)


def write_npz_cache(path: Path, X: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.setdefault("created_at", datetime.now().isoformat())
    np.savez_compressed(path, X=X, metadata=json.dumps(payload, sort_keys=True))


def read_npz_cache(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    return np.asarray(data["X"]), metadata


def compute_rem3di_from_zarr(
    dataset_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    progress_every_batches: int | None = 50,
) -> np.ndarray:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
    from threedscriptors.evaluation.regression.featurization import RemediDescriptorCalculator
    from threedscriptors.model.model_builder import ModelBuilder

    ds = MoleculeDataset.open_existing_dataset_from_dir(dataset_path, load_smiles=False)
    model = ModelBuilder.from_directory(str(checkpoint_path)).build_remedi_model()
    calc = RemediDescriptorCalculator(
        model,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        progress_every_batches=progress_every_batches,
    )
    return calc.calculate_descriptors(ds)


def rem3di_checkpoint_uses_structural_prior(checkpoint_path: Path) -> bool:
    from threedscriptors.model.model_builder import ModelBuilder

    model = ModelBuilder.from_directory(str(checkpoint_path)).build_remedi_model()
    return model.structural_prior is not None


def compute_ecfp_from_zarr(dataset_path: Path) -> np.ndarray:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

    ds = MoleculeDataset.open_existing_dataset_from_dir(dataset_path, load_smiles=True)
    return compute_ecfp(ds.get_smiles_per_structure())
