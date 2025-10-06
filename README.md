# REM3DI - REpresentation learning for Molecules with 3D Information


## Setup
1. pip install from pyproject.toml
2. Additionally clone and install a torch-sim fork from https://github.com/steffen-wedig/torch-sim


## Raw dataset download
...

## Compiling the a MoleculeDataset for Training
Find a number of example scripts in scripts/dataset_creation

MoleculeDatasets is the object that contains all embedding and structural data for our molecular dataset. All data is stored in zarr arrays (design motivitation: multithreaded read access from disk, when dataset size exceeds memory limit).
Batchwise compilation of datasets by: 

1. MoleculeGenerators yield batches of smiles or processed ase atoms.
2. Configurable Orchestrator will perform conformal sampling, relaxation or mace descriptor calculation.
3. Everything (Embeddings, structures, optional system wide or atom wise regression labels) will be shoved into a Molecule Dataset

## Running Pretraining
1. Create a training dir with a training_config.yaml and architecture_config.yaml **or** generate the architecture_config.yaml with the scripts/write_config_file.py script.
2. Run pretrainig via scripts/scripts/run_denoising_pretraining.py --train_dir "dir"



## Using Pretrained Model

from threedscriptors.model.model_builder import ModelBuilder
model = ModelBuilder.from_directory(model_dir).build_remedi_model()