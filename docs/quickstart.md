# Quickstart

**What you'll do:** load a published Rem3Di model and turn a small dataset of
molecules into descriptor vectors — in a few minutes.

## Prerequisites

- [Installed](installation.md) `remedi`.
- A **published model directory** (see [layout](concepts.md#model-directory)).
- A **MoleculeDataset** (zarr) to embed. Don't have one? See
  [Prepare a dataset](prepare-a-dataset.md), or use one of the example datasets.

## Get descriptors

`RemediCalculator` is the one-object way from a model directory to descriptors —
it loads the checkpoint and embeds a dataset for you.

```python
from remedi.evaluation.benchmark.descriptors import RemediCalculator
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset

# 1. Point the calculator at a published model directory.
calc = RemediCalculator(
    model_dir="/path/to/published_model",
    device="cuda",  # or "cpu"
)

# 2. Open a dataset and embed it.
dataset = MoleculeDataset.open_existing_dataset_from_dir(
    "/path/to/dataset_zarr"
)
descriptors = calc.calculate(dataset)   # numpy array (N_molecules, D)

print(descriptors.shape)
```

`descriptors` has one row per molecule — feed it straight into any downstream model.

> Running on a different machine than the model was trained on? The MACE path
> baked into the config may not exist locally — pass
> `RemediCalculator(model_dir=..., mace_model_path="/local/MACE.model")`. See
> [Concepts → Foundation-model frontend](concepts.md#foundation-model-frontend).

## Next steps

- [Train a downstream model](train-downstream.md) on these descriptors.
- [Evaluate a model](evaluate-a-model.md) across many benchmarks at once.
- Notebook: `examples/01_get_descriptors.ipynb`.
