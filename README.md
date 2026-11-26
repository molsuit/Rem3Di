# Rem3Di: REpresentation learning for Molecules with 3D Information

<div align="center">
  <b>Learning smooth, chiral 3D molecular representations from equivariant atomistic foundation models</b>
  <br> <br>
  <a href="https://www.linkedin.com/in/steffen-wedig/" target="_blank">Steffen&nbsp;Wedig</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.felixburton.com" target="_blank">Felix&nbsp;Burton</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.linkedin.com/in/rokas-elijošius-9148a8147/" target="_blank">Rokas&nbsp;Elijošius</a><sup>*</sup> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.phy.cam.ac.uk/profile/dr-christoph-schran/" target="_blank">Christoph&nbsp;Schran</a><sup>*</sup> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.larsschaaf.com" target="_blank">Lars&nbsp;L.&nbsp;Schaaf</a><sup>*</sup>
  <br> <br>
  <a href="https://arxiv.org/abs/2607.19977" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red"></a>
  &emsp;
  <a href="https://molsuit.github.io/Rem3Di/" target="_blank"><img alt="Documentation" src="https://img.shields.io/badge/Documentation-online-blue"></a>
  <br> <br>
</div>

<div align="center">
    <img width="600" alt="Rem3Di" src="https://raw.githubusercontent.com/molsuit/Rem3Di/b2574870bb4d10b00693b6af24661d25a30b9e84/docs/header-rem3di.jpg"/>
</div>


## Documentation

📖 Full usage docs (installation, quickstart, evaluating models, training
downstream heads, training from scratch, and concepts) are online at
**<https://molsuit.github.io/Rem3Di/>**.



## Setup

Install from a checkout with [uv](https://docs.astral.sh/uv/) (requires Python >=3.12). The committed `uv.lock` pins the workspace packages, the MACE fork and the CUDA 13.2 build of PyTorch. It covers Linux x86_64, and Linux aarch64 and Windows without the `cuda` extra; it does not cover macOS.

```bash
git clone https://github.com/molsuit/Rem3Di.git && cd Rem3Di
# PyTorch + MACE, descriptors on CPU or GPU, chemiscope exports, W&B logging
uv sync --all-packages --extra descriptor-analysis --extra telemetry
# the same plus the cuEquivariance kernels and nvmolkit (Linux x86_64 only)
uv sync --all-packages --extra descriptor-analysis --extra telemetry --extra cuda
```

See the [documentation](https://molsuit.github.io/Rem3Di/) for usage and full detail.

## Raw dataset download
We work with a number of different datasets for benchmarking Rem3Di.

Scripts to download these datasets from can be found in scripts/dataset_download. All scripts take the target destination folder as their first argument.

Example: Geom Drugs can be downloaded by:

```bash
mkdir raw_datasets
bash scripts/dataset_download/download_geomdrugs.sh ./raw_datasets
```

## Generating a Dataset for Training
TL;DR: a preparer writes each corpus once as a dataset (`scripts/dataset_creation/prepare_structures.py`), subsets of it are selections (`scripts/dataset_creation/select_structures.py`), and a pretraining config names a mixture of datasets and selections (`configs/training/mixture_smoke/` has two examples).


### Dataset Overview
A dataset is a directory with four files:

- `dataset.yaml`: the spec (`DatasetSpec`): labels, SMILES columns, evaluation settings.
- `table.parquet`: one row per structure, in `structure_id` order, with the ids (`structure_id`, `molecule_id`, `stereoisomer_id`), `total_charge`, `multiplicity`, the SMILES, the labels and the split columns.
- `structures.atp`: the frames, one [atompack](https://github.com/LeMaterial/atompack) record per table row (positions as float32, atomic numbers as uint8). The file is memory-mapped, so a single structure reads in microseconds and data-loader workers share it through the page cache.
- `provenance.yaml`: how the dataset was made, and its hashes.

`remedi_data.dataset.Dataset` (`packages/remedi-data/remedi_data/dataset/`) opens a dataset directory; `Dataset.read_structures()` returns `Structures`, the in-memory form (flat `positions` and `atomic_numbers` with a pointer `ptr`, plus `total_charge` and `multiplicity` from the table).

### Dataset Generation
Every corpus has a preparer under `packages/remedi-data/remedi_data/preparation/structures/sources/`. `scripts/dataset_creation/prepare_structures.py` runs one from a yaml config (one `kind` per source) and writes the dataset directory. SMILES-only benchmarks are prepared as bundles (a table without structures) and turned into datasets by `scripts/dataset_creation/build_datasets.py`, which generates the conformers. MACE descriptors are computed during training, not stored.

## Running Pretraining
1. Write a training config whose `data` section names a mixture of datasets and selections, next to an architecture config (examples in `configs/training/mixture_smoke/`).
2. Submit it to DAIS: `uv run python scripts/submit_job.py <training_config.yaml> --time 06:00:00`. Each config is frozen into its training directory and runs as one `scripts/pretrain.slurm` job on an H200 or B200. To run in the current shell instead: `uv run python scripts/run_online_embedding_denoising_pretraining.py --training_config <training_config.yaml>`.

Training reads structures through `TrainingMoleculeDataset` (`remedi/training/data/training_dataset.py`): each dataloader worker opens its own `StructureFile` on `structures.atp` and reads every sample from disk when it is asked for.


## Using Pretrained Model

```python
from pathlib import Path

from remedi.embedding import RemediConfig
from remedi_data.dataset import Dataset

calculator = RemediConfig(name="my_model", model_dir=Path(model_dir)).build()
dataset = Dataset.open(dataset_path)
descriptors = calculator.calculate(dataset)  # (structures, descriptor length)
```

To score a checkpoint on the benchmark panel or run the descriptor-space analyses (`configs/evaluation/descriptor_analysis/`), use `scripts/evaluation/embed_and_evaluate.py`.

## Citation

If you use parts of this code in your work, please cite our paper
(<a href="https://arxiv.org/abs/2607.19977" target="_blank">arXiv</a>), first released at the NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations. Arxiv:

<!-- > Wedig, S., Burton, F., Elijošius, R., Schran, C., & Schaaf, L. L. (2026). Rem3Di: Learning smooth, chiral 3D molecular descriptors from atomistic foundation models. arXiv preprint arXiv:2607.19977. -->

```bibtex
@article{wedig2026rem3di,
  title={REM3DI: Learning smooth, chiral 3D molecular representations from equivariant atomistic foundation models},
  author={Wedig, Steffen and Elijo{\v{s}}ius, Rokas and Schran, Christoph and Schaaf, Lars Leon},
  journal={arXiv preprint arXiv:2607.19977},
  year={2026},
  note={Previously released in 2025 at the NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations}
}
```
