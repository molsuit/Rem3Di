from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import MoleculeDatasetAnalysis

dataset_dir = Path("/scratch/s5f/wedigs.s5f/datasets/tmqm")

eval_dir = Path(
    "/home/s5f/wedigs.s5f/3DMolecularDescriptors/analysis_output"
)
dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
da = MoleculeDatasetAnalysis(dataset=dataset)
da.run()
da.output(output_dir=eval_dir)
