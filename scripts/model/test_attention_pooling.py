from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from 
from threedscriptors.evaluation.descriptor_analysis.clustering import UMAPCalculator
from threedscriptors.evaluation.descriptor_analysis.clustering_task import (
    DescriptorClusteringTask,
)
from threedscriptors.evaluation.evaluation_pipeline import EvalPipelineRunner
from threedscriptors.model.model_builder import ModelBuilder
from pathlib import Path


from threedscriptors.data_handling.sample import (
    PreprocessedSample,
    pretraining_padded_collate_fn,
)
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.data import (
    DatasetSplitting,
    worker_init_fn,
)
from torch.utils.data import DataLoader, Subset

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/pcqm_benchmark"
)


eval_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/pcqm_benchmark")

dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)


# Check the UMAP, descriptor statistics of the various aggregations of 
# 1. random atomic embeddings, mace atomic embeddings


from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)

eval_ds = TrainingMoleculeDataset.from_molecule_dataset(dataset, get_item = pos_emb_getitem)

training_loader = DataLoader(
        eval_ds,
        batch_size=128,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        num_workers=16,
        shuffle=True,
        collate_fn=pretraining_padded_collate_fn,
    )

for batch in training_loader:

    samples.to_(device)
    preprocessed_samples: PreprocessedSample = preprocessor(samples)



