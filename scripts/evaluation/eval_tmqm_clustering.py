from pathlib import Path

import torch

from threedscriptors.configuration.architecture_config import (
    EncoderOnlyArchitectureConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from threedscriptors.evaluation.descriptor_analysis import (
    CapacityDiagnosticTask,
    CoordinationNumberColor,
    DBlockColor,
    DescriptorAnalysisRunner,
    DescriptorDistributionTask,
    DescriptorNormalizationConfig,
    MetalCenterAtomicNumberColor,
    MetalCenterElementColor,
    NumAtomsColor,
    ProjectionConfig,
    ProjectionPlotTask,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)

MODEL_DIR = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/159-2025_08_19_14_29_24-tmqmpretrained"
)
DATASET_DIR = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/tmqm"
)
OUTPUT_DIR = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/tmqm_pretraining/train"
)


def load_model(model_dir: Path):
    model = EncoderOnlyArchitectureConfig.from_directory(str(model_dir)).build()
    model.encoder.load_state_dict(torch.load(model_dir / "encoder.pth"))
    model.preprocessor.atomic_preprocessor.load_state_dict(
        torch.load(model_dir / "atomic_preprocessor.pth")
    )
    model.preprocessor.geometric_preprocessor.load_state_dict(
        torch.load(model_dir / "geometric_preprocessor.pth")
    )
    return model


def main() -> None:
    model = load_model(MODEL_DIR)

    dataset = MoleculeDataset.open_existing_dataset_from_dir(DATASET_DIR)
    train_dataset = TrainingMoleculeDataset.from_molecule_dataset(
        dataset, get_item=atoms_getitem
    )

    descriptors = evaluate_molecular_descriptor_on_dataset(model, train_dataset)

    runner = DescriptorAnalysisRunner(
        normalization=DescriptorNormalizationConfig(z_score=False, l2_normalize=True),
        projection=ProjectionConfig(method="umap", center=True),
        tasks=[
            CapacityDiagnosticTask(),
            DescriptorDistributionTask(),
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
