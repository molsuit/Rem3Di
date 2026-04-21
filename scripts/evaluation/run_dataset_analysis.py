from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import MoleculeDatasetAnalysis

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs"
)

eval_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/dataset_analysis_geom_drugs"
)


dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)


da = MoleculeDatasetAnalysis(dataset=dataset)
da.run()
da.output(output_dir=eval_dir)
