# Concepts

Shared reference for the other pages: what a model directory contains, what a
dataset looks like, and the shape of a descriptor.

## Model directory

A trained/published model is a directory with four files:

| File | What it is |
|------|-----------|
| `post_training_architecture_config.yaml` | Architecture config saved by training (`kind: encoder_decoder`). |
| `encoder.pth` | Transformer encoder weights. |
| `atomic_preprocessor.pth` | Atomic-descriptor preprocessor (MACE normalization + projection). |
| `geometric_preprocessor.pth` | Geometric (pairwise-distance) preprocessor. |

Easiest is to let [`RemediCalculator`](evaluate-a-model.md#option-a-embed-a-dataset-python)
load it. To load the model directly, use `from_encoder_yaml`, which strips the
decoder block from the saved `encoder_decoder` config so the checkpoint loads for
inference:

```python
model = EncoderOnlyArchitectureConfig.from_encoder_yaml(model_dir).build()
model.encoder.load_state_dict(torch.load(model_dir / "encoder.pth"))
model.preprocessor.atomic_preprocessor.load_state_dict(
    torch.load(model_dir / "atomic_preprocessor.pth")
)
model.preprocessor.geometric_preprocessor.load_state_dict(
    torch.load(model_dir / "geometric_preprocessor.pth")
)
```

## Foundation-model frontend

Rem3Di builds on a frozen atomistic foundation model (a machine-learned
interatomic potential) that produces the atom-centred features it aggregates. The
framework is agnostic to the backbone; the published models use MACE. These
features are computed on the fly inside the model, and you do not precompute
them into the dataset. The foundation model to use is set by
`mace_config.model_path` in the architecture config. That path is absolute and
machine-specific, so when you move a checkpoint between machines, override it:

- **Python:** `RemediCalculator(model_dir=..., mace_model_path="/local/MACE.model")`
- **Manifest yaml:** `mace_model_path:` under the `model:` block.

## MoleculeDataset (zarr)

A dataset is a zarr directory. For embedding and pretraining it needs, per
molecule, the 3D structure. The base flow reads:

- `positions` `(N_atoms, 3)`: coordinates
- `atomic_numbers` `(N_atoms,)`
- `molecule_ptr` `(N_molecules+1,)`: atom-range pointers per molecule
- `total_charge`, `multiplicity` `(N_molecules,)`

Optional, for supervised/downstream work:

- `tasks/` group: regression/classification labels + validity masks, described
  by a `TaskSet` (`remedi/data_handling/dataset/tasks.py`).
- `split`: train/val/test codes.
- `smiles`: SMILES strings (needed for the ECFP baseline).

Open one with `TrainingMoleculeDataset(path, get_item=atoms_getitem)`. See
[Prepare a dataset](prepare-a-dataset.md) to build one.

## Descriptor shape

`evaluate_molecular_descriptor_on_dataset` returns `(N, L * d_out)`:

- `d_out`: the aggregator's output dim.
- `L`: number of pooled "seed" tokens, `1` for `mean`/`attention` aggregators,
  `num_seeds` for PMA. Rows are the flattened per-molecule descriptor.
