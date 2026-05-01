from pathlib import Path

import torch

from threedscriptors.configuration.architecture_config import (
    EncoderDecoderArchitectureConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from threedscriptors.model.remedi_model import REM3DIModel
from threedscriptors.evaluation.descriptor_analysis import (
    CapacityDiagnosticTask,
    CoordinationNumberColor,
    DBlockColor,
    DescriptorAnalysisRunner,
    DescriptorDistributionTask,
    DescriptorNormalizationConfig,
    HDBSCANClusterTask,
    MetalCenterAtomicNumberColor,
    MetalCenterElementColor,
    NumAtomsColor,
    ProjectionConfig,
    ProjectionPlotTask,
    TopNormDescriptorsTask,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)

MODEL_DIR = Path(
    "/scratch/s5f/wedigs.s5f/training_runs/10-2026_04_28_09_24_50-tmc_0"

)
DATASET_DIR = Path(
    "/scratch/s5f/wedigs.s5f/datasets/tmqm"
)
OUTPUT_DIR = Path(
    "/scratch/s5f/wedigs.s5f/training_runs/10-2026_04_28_09_24_50-tmc_0/analysis"
)


def load_model(model_dir: Path) -> REM3DIModel:
    bundle = EncoderDecoderArchitectureConfig.from_directory(str(model_dir)).build()
    model = REM3DIModel(preprocessor=bundle.preprocessor, encoder=bundle.encoder)
    model.encoder.load_state_dict(torch.load(model_dir / "encoder.pth"))
    model.preprocessor.atomic_preprocessor.load_state_dict(
        torch.load(model_dir / "atomic_preprocessor.pth")
    )
    model.preprocessor.geometric_preprocessor.load_state_dict(
        torch.load(model_dir / "geometric_preprocessor.pth")
    )
    return model


def main() -> None:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(DATASET_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors_path = OUTPUT_DIR / "descriptors.pt"

    if descriptors_path.exists():
        descriptors = torch.load(descriptors_path)
    else:
        model = load_model(MODEL_DIR)
        train_dataset = TrainingMoleculeDataset.from_molecule_dataset(
            dataset, get_item=atoms_getitem
        )
        descriptors = evaluate_molecular_descriptor_on_dataset(
            model, train_dataset, device="cuda"
        )
        torch.save(descriptors, descriptors_path)

    runner = DescriptorAnalysisRunner(
        normalization=DescriptorNormalizationConfig(z_score=False, l2_normalize=True),
        projection=ProjectionConfig(method="umap", center=True),
        tasks=[
            CapacityDiagnosticTask(),
            CapacityDiagnosticTask(
                l2_normalize=True,
                file_name="capacity_diagnostic_l2.yaml",
            ),
            CapacityDiagnosticTask(
                drop_outlier_quantile=0.99,
                file_name="capacity_diagnostic_no_outliers.yaml",
            ),
            DescriptorDistributionTask(),
            TopNormDescriptorsTask(n_top=10),
            ProjectionPlotTask(
                file_name="number_of_atoms.png",
                color_provider=NumAtomsColor(),
            ),
            ProjectionPlotTask(
                file_name="metal_center_element.png",
                color_provider=MetalCenterElementColor(),
            ),
            ProjectionPlotTask(
                file_name="metal_center_dblock.png",
                color_provider=MetalCenterAtomicNumberColor(),
            ),
            ProjectionPlotTask(
                file_name="metal_center_block.png",
                color_provider=DBlockColor(),
            ),
            ProjectionPlotTask(
                file_name="coordination_number.png",
                color_provider=CoordinationNumberColor(),
            ),
            *(
                HDBSCANClusterTask(
                    min_cluster_size=mcs,
                    figure_file_name=f"hdbscan_clusters_mcs{mcs}.png",
                    summary_file_name=f"hdbscan_clusters_mcs{mcs}.yaml",
                )
                for mcs in (50, 200, 1000, 5000)
            ),
            # ProjectionPlotTask(
            #    file_name="umap_homo_lumo_gap.png",
            #    color_provider=RegressionTargetColor(
            #        target_index=0, target_name="HOMO-LUMO gap"
            #    ),
            # ),
            # ChemiscopeProjectionTask(file_name="umap.json.gz"),
        ],
    )

    results = runner.run(descriptors=descriptors, dataset=dataset)
    runner.serialize(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
