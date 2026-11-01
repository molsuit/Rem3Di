"""Write precomputed-embeddings files from a Rem3Di checkpoint.

One embeddings pair per dataset (``BENCHMARK_DATA_FORMAT.md`` §7.5), read by
``remedi_eval``'s ``precomputed`` descriptor. A dataset whose pair already
matches its structures and the checkpoint is skipped. One dataset failing is
recorded and the rest are still embedded; the evaluation then reports the
missing embeddings for that dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from remedi.embedding.calculator import RemediConfig
from remedi_data.bundle import read_spec, structures_identity
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.embeddings_file import (
    producer_record,
    read_metadata,
    write_embeddings,
)

logger = logging.getLogger(__name__)


class EmbeddingFailure(BaseModel):
    dataset_id: str
    error: str


class EmbeddingReport(BaseModel):
    """What one :func:`embed_datasets` call did, per dataset id."""

    descriptor_name: str
    checkpoint_sha256: str
    written: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    failed: list[EmbeddingFailure] = Field(default_factory=list)


def embed_datasets(
    config: RemediConfig, dataset_dirs: list[Path], embeddings_root: Path
) -> EmbeddingReport:
    """Embed every dataset in ``dataset_dirs`` into ``embeddings_root``.

    Raises:
        FileNotFoundError: if the checkpoint directory holds no weights.
    """
    checkpoint_sha256 = config.identity()
    report = EmbeddingReport(
        descriptor_name=config.name, checkpoint_sha256=checkpoint_sha256
    )
    calculator = config.build()
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        dataset_id = dataset_dir.name
        try:
            dataset_id = read_spec(dataset_dir).dataset_id
            structures_sha256 = structures_identity(dataset_dir)
            existing = read_metadata(embeddings_root, dataset_id, config.name)
            if (
                existing is not None
                and existing.structures_sha256 == structures_sha256
                and existing.descriptor_identity == checkpoint_sha256
            ):
                logger.info("%s/%s is up to date", dataset_id, config.name)
                report.skipped.append(dataset_id)
                continue
            logger.info("embedding %s with %s", dataset_id, config.name)
            dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
            matrix = calculator.calculate(dataset)
            write_embeddings(
                embeddings_root,
                dataset_id=dataset_id,
                descriptor_name=config.name,
                descriptor_identity=checkpoint_sha256,
                structures_sha256=structures_sha256,
                structure_id=np.asarray(dataset.structure_ids[:], dtype=np.int64),
                embedding=matrix,
                producer=producer_record("remedi"),
            )
            report.written.append(dataset_id)
        except Exception as error:  # one dataset must not lose the others
            logger.exception("embedding %s failed; recording", dataset_id)
            report.failed.append(
                EmbeddingFailure(dataset_id=dataset_id, error=repr(error))
            )
    return report
