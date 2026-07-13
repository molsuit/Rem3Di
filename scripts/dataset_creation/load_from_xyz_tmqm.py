"""Build the tmQM MoleculeDataset (zarr v3 sharded).

tmQM ships one comment line per structure like::

    CSD_code = WELROW | q = 0 | S = 0 | Stoichiometry = ... | MND = 8 | ...

so the charge key is ``q`` and the spin key is ``S``. Crucially, ``S`` is
the *total spin* and tmQM stamps it 0 for every entry (DFT closed-shell
singlet -- the known tmQM quirk), NOT a multiplicity. Under the verbatim
multiplicity convention, feeding ``spin_key="S"`` would store multiplicity 0
(physically invalid). Leaving ``spin_key`` unset makes the generator default
to multiplicity 1.0 (closed-shell singlet), which is exactly correct for
every all-S=0 tmQM entry. (OMol25, by contrast, stores multiplicity in its
``spin`` field and is read verbatim.)
"""

from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from remedi.data_handling.dataset_creation.generators.xyz_generator import (
    XYZMoleculeGenerator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

xyz_files = [
    Path(f) for f in Path("/scratch/s5f/wedigs.s5f/raw_datasets/tmqm").glob("*.xyz")
]


generator = XYZMoleculeGenerator(
    xyz_file=xyz_files,
    loading_batch_size=10000,
    charge_key="q",
    spin_key=None,  # tmQM S is total spin, uniformly 0 -> multiplicity 1.0
)

filter_stage = FilterAtomsStage(
    config=FilterAtomsStageConfig(
        max_atoms=200,
        reject_zero_h=True,
        min_h_heavy_ratio=0.1,
    )
)
pipeline = [filter_stage, CopyDataStage(dtype=torch.float64)]

creation_config = DatasetCreationConfig(
    path=Path("/scratch/s5f/wedigs.s5f/datasets/tmqm"),
    N_structures=None,
)
# Defaults shard correctly; no per-chunk-file workaround needed under v3.
dataset_config = DatasetConfig(contains_smiles=False)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
