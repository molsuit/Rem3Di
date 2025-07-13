from pathlib import Path
from threedscriptors.data_handling.source_preprocessing.qm9_preprocessing import load_qm9
import numpy as np 

qm9_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/qm9_raw")




smiles, molecules, regression_targets, regression_masks, task_configs = load_qm9(qm9_dir)



print("SMILES:", smiles)
