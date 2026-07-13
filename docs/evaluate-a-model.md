# Evaluate a model

**What you'll do:** embed a dataset with a published model, or run a full
benchmark panel (many datasets × learners) against it.

## Prerequisites

- [Installed](installation.md) `remedi` (`--extra eval` for the panel).
- A [published model directory](concepts.md#model-directory).
- One or more [MoleculeDataset](concepts.md#moleculedataset-zarr) zarrs.

!!! tip "From SMILES — the quickest start"

    The fastest path: SMILES → 3D conformer → descriptor, no dataset files needed.
    `get_ase_atoms` runs RDKit ETKDG for one conformer per molecule; `embed_atoms`
    runs MACE and the Rem3Di encoder internally, so this is the full
    SMILES → MACE → descriptor chain:

    ```python
    from remedi.evaluation.benchmark.descriptors import RemediCalculator
    from remedi.data_handling.data_utils import get_ase_atoms

    calc = RemediCalculator(model_dir="/path/to/published_model", device="cuda")

    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    atoms = [get_ase_atoms(s) for s in smiles]
    X = calc.embed_atoms(atoms)  # (N, D)
    ```

    `embed_atoms(atoms, total_charge=0.0, multiplicity=1.0)` defaults to a neutral
    singlet for every molecule; override for ions/radicals. For relaxed or
    multi-conformer inputs use `get_ase_atoms_with_conformers(smiles, N)` (adds
    MMFF optimization; rejects charged molecules).

## Option A — embed a dataset (Python)

The direct route: model dir → descriptors. `RemediCalculator` loads the checkpoint
once, then embeds molecules from SMILES, from a prepared dataset, or from ASE
`Atoms` you build in memory. Create it once:

```python
from remedi.evaluation.benchmark.descriptors import RemediCalculator

calc = RemediCalculator(
    model_dir="/path/to/published_model",
    batch_size=64,
    device="cuda",
    # mace_model_path="/local/MACE.model",  # if not on this machine
)
```

The MACE weights are baked into the checkpoint config (`mace_config.model_path`),
so nothing extra is needed — unless that path doesn't exist on your machine, in
which case pass `mace_model_path` to point at your local copy.

### From SMILES

See the green **From SMILES — the quickest start** box near the top of this page:
build ASE `Atoms` with `get_ase_atoms`, then call `calc.embed_atoms(atoms)`.

### From a dataset

Embed a prepared [MoleculeDataset](concepts.md#moleculedataset-zarr) zarr on disk:

```python
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset

dataset = MoleculeDataset.open_existing_dataset_from_dir(
    "/path/to/dataset_zarr"
)
X = calc.calculate(dataset)  # (N, D)
```

### Custom dataloaders

Need the raw tensor or control over the dataloader (workers, prefetch)? Skip the
calculator, load the model yourself, and call
`evaluate_molecular_descriptor_on_dataset`. The loader is `from_encoder_yaml`,
which reads the encoder half of the saved config:

```python
import torch
from pathlib import Path
from remedi.configuration.architecture_config import (
    EncoderOnlyArchitectureConfig,
)
from remedi.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from remedi.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)

model_dir = Path("/path/to/published_model")
model = EncoderOnlyArchitectureConfig.from_encoder_yaml(model_dir).build()
model.encoder.load_state_dict(torch.load(model_dir / "encoder.pth"))
model.preprocessor.atomic_preprocessor.load_state_dict(
    torch.load(model_dir / "atomic_preprocessor.pth")
)
model.preprocessor.geometric_preprocessor.load_state_dict(
    torch.load(model_dir / "geometric_preprocessor.pth")
)
model.eval()

ds = TrainingMoleculeDataset("/path/to/dataset_zarr", get_item=atoms_getitem)
X = evaluate_molecular_descriptor_on_dataset(  # (N, L*d_out)
    model, ds, device="cuda", batch_size=64
)
```

## Option B — benchmark panel (CLI)

`run_eval.py` evaluates one model across many benchmark datasets and learners,
fault-tolerantly, writing results to disk. Write a manifest:

```yaml
# eval.yaml
model:
  descriptor_kind: remedi
  name: my-model            # cache id — use the checkpoint name
  model_dir: /path/to/published_model
  batch_size: 64
  device: cuda
  # mace_model_path: /local/MACE.model   # override if moved between machines

output_root: /path/to/eval_out
seed: 0

tasks:
  - kind: benchmark_panel
    eval_root: /path/to/benchmark_zarrs   # walked recursively for datasets
    learners:
      - learner_kind: linear
        ridge_alpha: 1.0
      - learner_kind: lightgbm
      - learner_kind: mlp
        hidden_dims: [256, 128]
        device: cuda
    baseline_descriptors:                 # optional comparison baselines
      - descriptor_kind: ecfp
        name: ecfp2048
        length: 2048
```

Run it:

```bash
uv run python scripts/evaluation/run_eval.py --config eval.yaml
```

## Outputs

- **Option A:** a `torch.Tensor` of shape `(N, L*d_out)`, [one row per molecule](concepts.md#descriptor-shape).
- **Option B:** under `output_root/` — benchmark `results.csv` (one row per
  dataset × descriptor × learner × target), a resolved `manifest.yaml`, and a
  `status.yaml` recording each task's success/failure.

## Next steps

- [Train a downstream model](train-downstream.md) on your own labels.
- Notebook: `examples/01_get_descriptors.ipynb`.
