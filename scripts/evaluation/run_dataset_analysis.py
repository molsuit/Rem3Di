from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import MoleculeDatasetAnalysis

dataset_dir = Path("/home/steffen/projects/mol_descriptors/dataset/tmqm")

eval_dir = Path(
    "/home/steffen/projects/mol_descriptors/eval_runs/dataset_analysis_tmqm"
)


dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)


da = MoleculeDatasetAnalysis(dataset=dataset)
da.run()
da.output(output_dir=eval_dir)
