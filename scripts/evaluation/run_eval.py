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
    ChemiscopeProjectionTask,
    DescriptorAnalysisRunner,
    DescriptorDistributionTask,
    DescriptorNormalizationConfig,
    NumAtomsColor,
    ProjectionConfig,
    ProjectionPlotTask,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)

DATASET_DIR = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs"
)
MODEL_DIR = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/geom_drugs_350k/1-2025_10_12_18_02_05-Train"
)
OUTPUT_DIR = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/geom_drugs_pretraining/geom_drugs"
)
MODEL_NAME = "GEOM_DRUGS"


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
        file_prefix=MODEL_NAME,
        normalization=DescriptorNormalizationConfig(z_score=True, l2_normalize=True),
        projection=ProjectionConfig(method="umap"),
        tasks=[
            CapacityDiagnosticTask(),
            DescriptorDistributionTask(),
            ProjectionPlotTask(
                file_name="umap_num_atoms.png",
                color_provider=NumAtomsColor(),
            ),
            ChemiscopeProjectionTask(file_name="umap.json.gz"),
        ],
    )

    results = runner.run(descriptors=descriptors, dataset=dataset)
    runner.serialize(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
