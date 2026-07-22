# Prepare a dataset

This page turns SMILES or 3D structures into a **MoleculeDataset** zarr, the
input format for [evaluation](evaluate-a-model.md) and
[training](train-from-scratch.md).

You do not precompute MACE features; the model computes them on the fly from
coordinates ([Concepts](concepts.md#foundation-model-frontend)). A dataset just needs 3D
structures (and optionally labels).

## Prerequisites

- [Installed](installation.md) `remedi`.
- SMILES (one per line), or structures (xyz/sdf).

## From SMILES

Datasets are built by a `DatasetConstructionOrchestrator` running a pipeline of
stages (filter, generate conformers, copy data) over a molecule generator.
`scripts/dataset_creation/load_from_smiles.py` is a ready template; copy it and
edit the input/output paths:

```python
import torch
from pathlib import Path
from remedi.configuration.dataset_config import (
    DatasetConfig, DatasetCreationConfig, FilterMoleculeStageConfig,
)
from remedi.data_handling.dataset_creation.generators.smiles_list_generator import (
    SmilesMoleculeGenerator, open_smiles_file,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    ConformerGenerationStage, CopyDataStage, FilterMoleculeStage,
)

smiles = open_smiles_file(Path("my_molecules.smi"))   # one SMILES per line
creation_config = DatasetCreationConfig(
    path=Path("./my_dataset"), N_structures=len(smiles)
)

gen = SmilesMoleculeGenerator(smiles, batch_size=500)
pipeline = [
    FilterMoleculeStage(config=FilterMoleculeStageConfig(max_atoms=100)),
    ConformerGenerationStage(dataset_creation_config=creation_config),
    CopyDataStage(dtype=torch.float64),
]
dataset_config = DatasetConfig(
    atom_chunk=450, molecule_chunk=50, contains_smiles=True, tasks=None
)

DatasetConstructionOrchestrator(
    pipeline=pipeline, batch_generator=gen,
    construction_config=creation_config, dataset_config=dataset_config,
).build_dataset()
```

## From existing structures

Use the matching template scripts, same orchestrator pattern:

- `scripts/dataset_creation/load_from_xyz.py`: xyz structures
- `scripts/dataset_creation/load_from_sdf_structures.py`: sdf
- `scripts/dataset_creation/load_qm9.py`: QM9

Dataset download helpers live in `scripts/dataset_download/`.

## Adding labels (for downstream training)

To attach regression/classification targets, set `tasks` on the `DatasetConfig`
to a `TaskSet` describing your label columns
(`remedi/data_handling/dataset/tasks.py`) and write the target values
into the dataset's `tasks/` group. The
[benchmark panel](train-downstream.md#option-b-from-a-labelled-dataset-cli) then
reads labels and splits automatically.

## Outputs

A zarr directory (`positions`, `atomic_numbers`, `molecule_ptr`, `total_charge`,
`multiplicity`, optional `smiles`/`tasks`/`split`); see
[Concepts](concepts.md#moleculedataset-zarr). Open it with
`TrainingMoleculeDataset(path, get_item=atoms_getitem)`.

## Next steps

- [Evaluate a model](evaluate-a-model.md) on your dataset.
- [Train from scratch](train-from-scratch.md) using it.
- Notebook: `examples/03_build_dataset_from_smiles.ipynb`.
