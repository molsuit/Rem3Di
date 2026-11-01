"""ECFP fingerprints over a dataset's SMILES, via molfeat."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel

from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.base import DescriptorCalculator


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


class EcfpConfig(BaseModel):
    descriptor_kind: Literal["ecfp"] = "ecfp"
    # Human-readable identifier used in the cache filename. Change this if you
    # change `fingerprint` or `length` so different ECFP variants don't clash.
    name: str = "ecfp"
    fingerprint: str = "ecfp"
    length: int = 2048

    def build(self) -> EcfpCalculator:
        return EcfpCalculator(fingerprint=self.fingerprint, length=self.length)

    def identity(self) -> str:
        """sha256 over the config: every knob that changes the matrix (§7c).

        ``name`` is a label the user picks and is free to reuse, so it cannot
        be the cache key on its own — ``fingerprint`` and ``length`` decide the
        numbers and both ride in here.
        """
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()
