from pathlib import Path

from threedscriptors.configuration.dataset_analysis_config import (
    BitBirchConfig,
    MoleculeDatasetAnalysisConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset_analysis import MoleculeDatasetAnalysis

dataset_dir = Path("/p/scratch/mace/wedig1/datasets/qm9")

eval_dir = Path(
    "/p/project1/mace/wedig1/3DMolecularDescriptors/analysis_output/qm9"
)

# Let BitBIRCH derive its threshold from the data (mean iSIM + factor*std via
# bblean.guess_threshold) instead of the hard-coded 0.65.
config = MoleculeDatasetAnalysisConfig(
    bitbirch=BitBirchConfig(auto_threshold=True)
)

dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
da = MoleculeDatasetAnalysis(dataset=dataset, config=config)
da.run()
da.output(output_dir=eval_dir)
