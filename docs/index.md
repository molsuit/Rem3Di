# Rem3Di documentation

<div align="center">
  <b>Learning smooth, chiral 3D molecular representations from equivariant atomistic foundation models</b>
  <br><br>
  Steffen&nbsp;Wedig &emsp;<b>&middot;</b>&emsp;
  Felix&nbsp;Burton &emsp;<b>&middot;</b>&emsp;
  Rokas&nbsp;Elijošius<sup>*</sup> &emsp;<b>&middot;</b>&emsp;
  Christoph&nbsp;Schran<sup>*</sup> &emsp;<b>&middot;</b>&emsp;
  Lars&nbsp;L.&nbsp;Schaaf<sup>*</sup>
  <br><br>
  <b>Paper:</b>
  <a href="https://arxiv.org/abs/2607.19977" target="_blank"><b><u>arXiv (2026)</u></b></a> &emsp;<b>&middot;</b>&emsp;
  <a href="https://openreview.net/forum?id=jOmZsvXoK5&referrer" target="_blank"><b><u>NeurIPS Workshop 2025</u></b></a>
  <br><br>
</div>

**Rem3Di** (`remedi`) repurposes the latent features of a frozen atomistic
foundation model (a machine-learned interatomic potential such as MACE) into a
single fixed-length descriptor of a whole molecule. The descriptor reflects the
molecule's three-dimensional shape and does not depend on the order in which the
atoms are listed. To capture handedness it adds **pseudoscalar** features, which
are unchanged by rotation but reverse sign under mirror reflection, so the
descriptor distinguishes enantiomers. The result is a feature vector you can use
directly for property prediction, virtual screening, and retrieval.

<p align="center">
  <img src="header-rem3di.jpg" alt="Rem3Di" width="600"/>
</p>

<p align="center">
  <strong>Paper:</strong>
  <a href="https://openreview.net/challenge?redirect=%2Fpdf%3Fid%3DjOmZsvXoK5" target="_blank" rel="noopener">NeurIPS 2025 Workshop</a>
  &middot; arXiv (coming soon)
</p>

## How to read these docs

Pick the page for what you want to do. Each page follows the same layout: what
you'll do, prerequisites, steps, outputs, and next steps.

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

Most users only need the Evaluate and Train-downstream flows: take a published
model, embed your molecules, and fit a head on your labels. Training from scratch
is for producing a new Rem3Di descriptor model.

## Runnable examples

Short, copy-and-adapt notebooks live in [`examples/`](https://github.com/molsuit/Rem3Di/tree/main/examples):

- `01_get_descriptors.ipynb`: model dir to descriptors
- `02_train_downstream_head.ipynb`: descriptors + labels to a trained head (runs on synthetic data, no GPU)
- `03_build_dataset_from_smiles.ipynb`: SMILES to a MoleculeDataset
- `04_pretrain_mini.ipynb`: a smoke-sized pretraining run
- `05_pseudoscalars.ipynb`: equivariant features to chirality-sensitive pseudoscalars (CPU, no model)
