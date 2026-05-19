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
) -> np.ndarray:
    """Extract REM3DI descriptors for every structure in a native zarr.

    Uses macepolar's canonical ``RemediDescriptorCalculatorConfig`` so the
    descriptors produced here are bit-identical to the ones Steffen's
    regression pipeline produces from the same checkpoint — there is no
    second, drift-prone reimplementation of the model loader in EVAL-001.

    ``checkpoint_path`` is a macepolar training-run directory containing
    ``post_training_architecture_config.yaml`` plus ``encoder.pth``,
    ``atomic_preprocessor.pth`` and ``geometric_preprocessor.pth`` (the
    layout written by macepolar's training entrypoints). Runs on CUDA:
    macepolar's ``evaluate_molecular_descriptor_on_dataset`` defaults to
    ``device="cuda"`` and does not parameterise the device, matching the
    regression-eval path — produce REM3DI caches on a GPU host.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
    from threedscriptors.evaluation.regression.featurization import (
        RemediDescriptorCalculatorConfig,
    )

    ds = MoleculeDataset.open_existing_dataset_from_dir(Path(dataset_path))
    calc = RemediDescriptorCalculatorConfig(
        model_dir=Path(checkpoint_path)
    ).get_descriptor_calculator()
    return np.asarray(calc.calculate_descriptors(ds), dtype=np.float32)


def compute_ecfp_from_zarr(dataset_path: Path) -> np.ndarray:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

    ds = MoleculeDataset.open_existing_dataset_from_dir(Path(dataset_path))
    return compute_ecfp(ds.get_smiles_per_structure())
