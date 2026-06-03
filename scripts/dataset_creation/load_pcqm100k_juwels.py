"""Build a fixed-size pcqm4m structure subset on JUWELS for fast iteration.

Mirrors load_from_sdf_structures.py but with /p/scratch paths. The
SDFMoleculeGenerator stops after the orchestrator has accepted
``--n-structures`` rows, so this only ever reads the first chunk of the 3.5M
molecule SDF -- much faster than the full build. The subset size is a CLI
argument (default 100k, kept for the original behaviour); the output path
defaults to ``pcqm<N>k_std`` under the datasets dir.

Run with (e.g. a 500k subset):
    uv run python scripts/dataset_creation/load_pcqm100k_juwels.py \\
        --n-structures 500000
"""

import argparse
from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
    FilterMoleculeStageConfig,
    PhysicochemicalDescriptorStageConfig,
)
from threedscriptors.data_handling.dataset.tasks import ElementSet
from threedscriptors.data_handling.dataset_creation.generators.sdf_generator import (
    SDFMoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
    PhysicochemicalDescriptorStage,
)

sdf_file = Path(
    "/p/scratch/mace/wedig1/raw_datasets/pcqm4m/extracted/pcqm4m-v2-train.sdf"
)
datasets_dir = Path("/p/scratch/mace/wedig1/datasets/pcqm4m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-structures",
        type=int,
        default=100_000,
        help="Number of accepted structures to keep (default: 100000).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output dataset directory "
        "(default: <datasets>/pcqm<N>k_std derived from --n-structures).",
    )
    return parser.parse_args()


def default_output_path(n_structures: int) -> Path:
    return datasets_dir / f"pcqm{n_structures // 1000}k_std"


def main() -> None:
    args = parse_args()
    output_path = (
        args.output_path
        if args.output_path is not None
        else default_output_path(args.n_structures)
    )

    # Match the benchmark filter config (configs/dataset_creation/benchmarks_local.yaml)
    # so pretrain SMILES go through the same canonicalization as eval SMILES.
    smiles_filter = FilterMoleculeStageConfig(
        max_atoms=100,
        element_set=ElementSet.mace_off,
        strip_salts=True,
        neutralize=True,
        dedupe=True,
    )

    mol_generator = SDFMoleculeGenerator(
        sdf_file=sdf_file,
        loading_batch_size=10000,
        filter_config=smiles_filter,
    )

    filter_stage = FilterAtomsStage(
        config=FilterAtomsStageConfig(element_set=ElementSet.mace_off)
    )
    # Cheap RDKit physicochemical descriptors -> per-structure targets_system
    # columns, used as linear-probe labels during pretraining. Runs after the
    # atoms filter (so it sees the loaded 3D structures + SMILES) and before the
    # copy stage.
    physchem_cfg = PhysicochemicalDescriptorStageConfig()
    physchem_stage = PhysicochemicalDescriptorStage(config=physchem_cfg)
    copy_data = CopyDataStage(dtype=torch.float64)

    creation_config = DatasetCreationConfig(
        path=output_path,
        N_structures=args.n_structures,
    )
    dataset_config = DatasetConfig(tasks=physchem_cfg.to_task_set())

    orchestrator = DatasetConstructionOrchestrator(
        pipeline=[filter_stage, physchem_stage, copy_data],
        batch_generator=mol_generator,
        construction_config=creation_config,
        dataset_config=dataset_config,
    )

    orchestrator.build_dataset()


if __name__ == "__main__":
    main()
