"""Embeddings computed elsewhere, read from an embeddings pair per dataset.

Every trained encoder (Rem3Di, UniMol2, MuMo, ...) enters the evaluation this
way; ``remedi_eval`` never imports a model. The pairs are written by the
model's own package (for Rem3Di, ``remedi.embedding``) with
:func:`remedi_eval.descriptors.embeddings_file.write_embeddings`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.embeddings_file import (
    embeddings_aligned_to,
    read_embeddings,
)


class PrecomputedEmbeddingsConfig(BaseModel):
    """Read ``<embeddings_root>/<dataset_id>/<name>.{npz,yaml}`` for each dataset."""

    model_config = ConfigDict(extra="forbid")

    descriptor_kind: Literal["precomputed"] = "precomputed"
    name: str
    embeddings_root: Path

    def identity(self) -> str:
        """Within one run, the files are named by the root and the name."""
        key = {
            "name": self.name,
            "embeddings_root": str(self.embeddings_root.resolve()),
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

    def load_matrix(
        self, dataset: MoleculeDataset, dataset_id: str, cache_root: Path
    ) -> np.ndarray:
        """The embeddings in zarr order; ``cache_root`` is unused (the files are the cache).

        Raises:
            FileNotFoundError: if there are no embeddings for this dataset.
            EmbeddingsFileError: if they were computed on other data or do not
                cover exactly the dataset's structures.
        """
        del cache_root
        embeddings = read_embeddings(self.embeddings_root, dataset_id, self.name)
        return embeddings_aligned_to(
            embeddings,
            Path(dataset.path),
            np.asarray(dataset.structure_ids[:], dtype=np.int64),
        )
