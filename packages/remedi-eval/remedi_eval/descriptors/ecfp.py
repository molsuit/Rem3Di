"""ECFP fingerprints over a dataset's SMILES (:mod:`remedi_data.chemistry.fingerprints`).

The matrix is cached as an embeddings pair (:mod:`embeddings_file`) at
``<cache_root>/<dataset_id>/<name>.npz`` with the config's hash as
``descriptor_identity``. A pair whose ``structures_sha256`` or identity does
not match the request, or that does not read back intact, is recomputed and
overwritten (:func:`~remedi_eval.descriptors.embeddings_file.ensure_embeddings`,
under a per-pair file lock), so one name holds one variant at a time and a
stale matrix is never served.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from remedi_data.chemistry.fingerprints import morgan_fingerprints
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.embeddings_file import ensure_embeddings, producer_record


class EcfpConfig(BaseModel):
    """Morgan bits as a descriptor; radius 2 is ECFP4."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ecfp"] = "ecfp"
    # Human-readable identifier used in the cache filename. Change this if you
    # change `radius` or `fingerprint_size` so different ECFP variants don't clash.
    name: str = "ecfp"
    radius: int = Field(2, ge=0)
    fingerprint_size: int = Field(2048, gt=0)

    def identity(self) -> str:
        """sha256 over the config: every knob that changes the matrix.

        ``name`` is a label the user picks and is free to reuse, so it cannot
        be the cache key on its own — ``radius`` and ``fingerprint_size`` decide
        the numbers and both ride in here.
        """
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def calculate(self, dataset: MoleculeDataset) -> np.ndarray:
        """``(N, fingerprint_size)`` float32 bits, one row per structure."""
        bits = morgan_fingerprints(
            dataset.get_smiles_per_structure(),
            layout="dense",
            radius=self.radius,
            size=self.fingerprint_size,
        )
        return bits.astype(np.float32)

    def load_matrix(
        self, dataset: MoleculeDataset, dataset_id: str, cache_root: Path
    ) -> np.ndarray:
        """The fingerprint matrix in zarr order, cached under ``cache_root``.

        Raises:
            FileNotFoundError: if the dataset directory carries no
                ``provenance.yaml`` to identify its contents by.
        """
        return ensure_embeddings(
            cache_root,
            dataset,
            dataset_id=dataset_id,
            descriptor_name=self.name,
            descriptor_identity=self.identity(),
            compute=lambda: self.calculate(dataset),
            producer=producer_record("remedi-eval"),
        ).embedding
