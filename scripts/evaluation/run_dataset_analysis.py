from pathlib import Path

from remedi.configuration.dataset_analysis_config import (
    BitBirchConfig,
    MoleculeDatasetAnalysisConfig,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_analysis import MoleculeDatasetAnalysis

dataset_dir = Path("/path/to/datasets/qm9")

eval_dir = Path("/path/to/3DMolecularDescriptors/analysis_output/qm9")

# Let BitBIRCH derive its threshold from the data (mean iSIM + factor*std via
# bblean.guess_threshold) instead of the hard-coded 0.65.
config = MoleculeDatasetAnalysisConfig(bitbirch=BitBirchConfig(auto_threshold=True))

dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)
da = MoleculeDatasetAnalysis(dataset=dataset, config=config)
da.run()
da.output(output_dir=eval_dir)
