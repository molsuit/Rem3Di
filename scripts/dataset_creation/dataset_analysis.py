from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import MoleculeDatasetAnalysis

dir = Path("/local/data/public/snw30/geom_drugs")

out_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/dataset_analysis_geom_drugs"
)
dataset = MoleculeDataset.open_existing_dataset_from_dir(dir)


da = MoleculeDatasetAnalysis(dataset)

da.run()
da.output(out_dir)
