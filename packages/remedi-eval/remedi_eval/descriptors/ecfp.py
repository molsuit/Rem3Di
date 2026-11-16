"""ECFP fingerprints over a dataset's SMILES (:mod:`remedi_data.chemistry.fingerprints`)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from remedi_data.chemistry.fingerprints import morgan_fingerprints
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.base import DescriptorCalculator
from remedi_eval.descriptors.cache import compute_and_cache


class EcfpCalculator(DescriptorCalculator):
    def __init__(self, *, radius: int, fingerprint_size: int) -> None:
        self.radius = radius
        self.fingerprint_size = fingerprint_size

    def calculate(self, dataset: MoleculeDataset) -> np.ndarray:
        bits = morgan_fingerprints(
            dataset.get_smiles_per_structure(),
            layout="dense",
            radius=self.radius,
            size=self.fingerprint_size,
        )
        return bits.astype(np.float32)


class EcfpConfig(BaseModel):
    """Morgan bits as a descriptor; radius 2 is ECFP4."""

    model_config = ConfigDict(extra="forbid")

    descriptor_kind: Literal["ecfp"] = "ecfp"
    # Human-readable identifier used in the cache filename. Change this if you
    # change `radius` or `fingerprint_size` so different ECFP variants don't clash.
    name: str = "ecfp"
    radius: int = Field(2, ge=0)
    fingerprint_size: int = Field(2048, gt=0)

    def build(self) -> EcfpCalculator:
        return EcfpCalculator(
            radius=self.radius, fingerprint_size=self.fingerprint_size
        )

    def identity(self) -> str:
        """sha256 over the config: every knob that changes the matrix.

        ``name`` is a label the user picks and is free to reuse, so it cannot
        be the cache key on its own — ``radius`` and ``fingerprint_size`` decide
        the numbers and both ride in here.
        """
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def load_matrix(
        self, dataset: MoleculeDataset, dataset_id: str, cache_root: Path
    ) -> np.ndarray:
        """The fingerprint matrix in zarr order, cached under ``cache_root``."""
        return compute_and_cache(self, dataset, cache_root, dataset_id)
