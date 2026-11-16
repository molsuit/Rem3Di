"""Write precomputed-embeddings files from a Rem3Di checkpoint.

One embeddings pair per dataset (``BENCHMARK_DATA_FORMAT.md`` §7.5), read by
``remedi_eval``'s ``precomputed`` descriptor. Each pair goes through
:func:`~remedi_eval.descriptors.embeddings_file.ensure_embeddings`, so a dataset
whose pair already matches its structures and the checkpoint identity is
skipped. One dataset failing is recorded and the rest are still embedded; the
evaluation then reports the missing embeddings for that dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from remedi.embedding.calculator import RemediConfig
from remedi_data.bundle import read_spec
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_eval.descriptors.embeddings_file import ensure_embeddings, producer_record

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
    producer = producer_record("remedi")
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        dataset_id = dataset_dir.name
        try:
            dataset_id = read_spec(dataset_dir).dataset_id
            dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
            result = ensure_embeddings(
                embeddings_root,
                dataset,
                dataset_id=dataset_id,
                descriptor_name=config.name,
                descriptor_identity=checkpoint_sha256,
                compute=lambda dataset=dataset: calculator.calculate(dataset),
                producer=producer,
            )
            if result.computed:
                report.written.append(dataset_id)
            else:
                report.skipped.append(dataset_id)
        except Exception as error:  # one dataset must not lose the others
            logger.exception("embedding %s failed; recording", dataset_id)
            report.failed.append(
                EmbeddingFailure(dataset_id=dataset_id, error=repr(error))
            )
    return report
