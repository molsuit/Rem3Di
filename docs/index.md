# Rem3Di documentation

**Rem3Di** (`remedi`) turns a 3D molecular structure into a fixed-length
descriptor vector, using an equivariant atomistic foundation model (MACE) as a
frontend. Those descriptors are drop-in features for property prediction,
retrieval, clustering, and other downstream tasks.

<p align="center">
  <img src="header-rem3di.jpg" alt="Rem3Di" width="600"/>
</p>

## How to read these docs

Pick the page for what you want to do. Each page is self-contained: **what you'll
do → prerequisites → steps → outputs → next steps**.

| I want to…                                                          | Start here                                      |
| ------------------------------------------------------------------- | ----------------------------------------------- |
| Install the package                                                 | [Installation](installation.md)                 |
| Get descriptors from a published model in 5 minutes                 | [Quickstart](quickstart.md)                     |
| Embed a dataset / benchmark a model                                 | [Evaluate a model](evaluate-a-model.md)         |
| Train a predictor on my own labels                                  | [Train a downstream model](train-downstream.md) |
| Train a Rem3Di model from scratch                                   | [Train from scratch](train-from-scratch.md)     |
| Turn SMILES/structures into a dataset                               | [Prepare a dataset](prepare-a-dataset.md)       |
| Extract chirality-sensitive pseudoscalars from equivariant features | [Pseudoscalars](pseudoscalars.md)               |
| Understand model dirs, datasets, descriptor shapes                  | [Concepts](concepts.md)                         |

## The three stages

```
                     ┌─────────────────────────────────────────────┐
 SMILES / xyz  ──►   │ Prepare a dataset  →  MoleculeDataset (zarr) │
                     └─────────────────────────────────────────────┘
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                             ▼
   Evaluate a model          Train a downstream model     Train from scratch
   (published .pth →         (frozen descriptors +        (self-supervised
    descriptors)              your labels → head)          denoising pretrain)
```

Most users only need **Evaluate** + **Train downstream**: take a published model,
embed your molecules, fit a head on your labels. Training from scratch is for
producing a new foundation descriptor.

## Runnable examples

Short, copy-and-adapt notebooks live in [`examples/`](https://github.com/steffen-wedig/3DMolecularDescriptors/tree/main/examples):

- `01_get_descriptors.ipynb` — model dir → descriptors
- `02_train_downstream_head.ipynb` — descriptors + labels → trained head (runs on synthetic data, no GPU)
- `03_build_dataset_from_smiles.ipynb` — SMILES → MoleculeDataset
- `04_pretrain_mini.ipynb` — a smoke-sized pretraining run
- `05_pseudoscalars.ipynb` — equivariant features → chirality-sensitive pseudoscalars (CPU, no model)

## Building this site

These pages are plain Markdown. To render them as an HTML site:

```bash
pip install mkdocs-material
mkdocs serve   # http://127.0.0.1:8000
```
