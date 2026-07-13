"""Build the curated/recalculated QM9 dataset from its extxyz dump.

Reads the single curated QM9 extxyz (one frame per molecule, the 15 GDB-9
properties + reference SMILES on each comment line) and writes a zarr dataset
with all 15 properties as system-scope regression tasks. No train/val/test
split is baked in -- splitting is left to train time.

Run with:
    uv run python scripts/dataset_creation/load_qm9.py
"""

from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from remedi.data_handling.dataset.tasks import ElementSet
from remedi.data_handling.dataset_creation.generators.qm9_generator import (
    ALL_QM9_TASKS,
    QM9Generator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

qm9_extxyz = Path(
    "/path/to/raw_datasets/recalcQM9/curatedQM9_full.extxyz"
)
output_path = Path("/path/to/datasets/qm9")


def main() -> None:
    generator = QM9Generator(
        xyz_file=qm9_extxyz,
        tasks=ALL_QM9_TASKS,
        loading_batch_size=10000,
    )

    # QM9 is GDB-9: small neutral organics over the MACE-OFF element set. The
    # atoms-side filter is the only gate we need (the generator already drops
    # rows whose SMILES RDKit rejects).
    filter_stage = FilterAtomsStage(
        config=FilterAtomsStageConfig(element_set=ElementSet.mace_off)
    )
    copy_data = CopyDataStage(dtype=torch.float64)

    creation_config = DatasetCreationConfig(path=output_path, N_structures=None)
    dataset_config = DatasetConfig(tasks=generator.task_set())

    orchestrator = DatasetConstructionOrchestrator(
        pipeline=[filter_stage, copy_data],
        batch_generator=generator,
        construction_config=creation_config,
        dataset_config=dataset_config,
    )

    orchestrator.build_dataset()


if __name__ == "__main__":
    main()
