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
  <a href="https://openreview.net/forum?id=jOmZsvXoK5&referrer" target="_blank"><b><u>Paper Link</u></b></a>
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

Install with [uv](https://docs.astral.sh/uv/) (requires Python >=3.12):

```bash
uv pip install "remedi[cpu]"    # PyTorch + MACE, CPU descriptors
uv pip install "remedi[cuda]"   # adds GPU accelerators
```

Or with pip:

```bash
pip install "remedi[cpu]"
```

See the [installation docs](https://molsuit.github.io/Rem3Di/) for extras, GPU builds, and full detail.

## Raw dataset download
We work with a number of different datasets for benchmarking Rem3Di.

Scripts to download these datasets from can be found in scripts/dataset_download. All scripts take the target destination folder as their first argument.

Example: Geom Drugs can be downloaded by:

```bash
mkdir raw_datasets
bash scripts/dataset_download/download_geomdrugs.sh ./raw_datasets
```

## Generating a MoleculeDataset for Training
TL;DR: We generate molecular datasets containing molecular structures and their mace descriptors from various sources, examples can be found in ```scripts/dataset_creation```.


### Dataset Overview
MoleculeDatasets (```remedi/data_handling/dataset/molecule_dataset.py```) is the object that contains all embedding and structural data for our molecular datasets. The datasets can further contain system wide, molecular labels, or atomwise labels, which can be used for training regression models. Our datasets are backed by zarr arrays(Design motivitation: multithreaded read access from disk, when dataset size exceeds memory limit). As the size of molecules varies, the MoleculeDataset contain pointer values that point to the molecules atom limits.

### Dataset Generation
Dataset generation is implemented batchwise to increase performance, as GPU memory can be fully utilized in this way. The pipeline goes through the following stages.

1. MoleculeGenerators yield batches of smiles or processed ase atoms, in case the molecular structures are available in the datasets. For a new data source, like a new dataset that stores molecular data in its own way, you need to implement a new MoleculeGenerator.
2. Configurable Orchestrator will execute the pipeline. E.g. perform conformal sampling, relaxation. Lastly, the mace descriptors will be configured
3. Everything (Embeddings, structures, optional system wide or atom wise regression labels) will be shoved into the right zarr arrays in the MoleculeDataset.

## Running Pretraining
1. Create a training dir with a training_config.yaml and architecture_config.yaml **or** generate the architecture_config.yaml with the scripts/write_config_file.py script. In the train dir
2. Run pretrainig via scripts/scripts/run_denoising_pretraining.py --train_dir "dir"

One note re. training: For training we create new dataset object (class TrainingMoleculeDataset). This contains additionally logic how multiple dataloader workers can simulatounsly retireve data from the same zarr files.


## Using Pretrained Model



```python
import torch
from remedi.configuration.architecture_config import EncoderOnlyArchitectureConfig
from remedi.evaluation.evaluation_utils import evaluate_molecular_descriptor_on_dataset
from remedi.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
ds = TrainingMoleculeDataset(training_config.dataset_path, get_item=pos_emb_getitem)
model = EncoderOnlyArchitectureConfig.from_directory(model_dir).build()
model.encoder.load_state_dict(torch.load(f"{model_dir}/encoder.pth"))
model.preprocessor.atomic_preprocessor.load_state_dict(
    torch.load(f"{model_dir}/atomic_preprocessor.pth")
)
model.preprocessor.geometric_preprocessor.load_state_dict(
    torch.load(f"{model_dir}/geometric_preprocessor.pth")
)
remedi_descriptors = evaluate_molecular_descriptor_on_dataset(model, ds)
```

## Citation

If you use any of this code in your work, please cite our paper
(<a href="https://openreview.net/forum?id=jOmZsvXoK5&referrer" target="_blank">OpenReview</a>):

> Wedig, Steffen, et al. <a href="https://openreview.net/forum?id=jOmZsvXoK5&referrer" target="_blank">"REM3DI: Learning smooth, chiral 3D molecular representations from equivariant atomistic foundation models."</a> NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations. 2025.

```bibtex
@inproceedings{wedig2025rem3di,
  title={REM3DI: Learning smooth, chiral 3D molecular representations from equivariant atomistic foundation models},
  author={Wedig, Steffen and Elijo{\v{s}}ius, Rokas and Schran, Christoph and Schaaf, Lars Leon},
  booktitle={NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations},
  year={2025}
}
```
