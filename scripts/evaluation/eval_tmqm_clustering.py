import argparse
from pathlib import Path

import torch

from remedi.configuration.architecture_config import (
    EncoderDecoderArchitectureConfig,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from remedi.evaluation.descriptor_analysis import (
    CapacityDiagnosticTask,
    CoordinationNumberColor,
    DBlockColor,
    DescriptorAnalysisRunner,
    DescriptorNormalizationConfig,
    MetalCenterAtomicNumberColor,
    MetalCenterElementColor,
    NumAtomsColor,
    ProjectionConfig,
    ProjectionPlotTask,
)
from remedi.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from remedi.model.remedi_model import REM3DIModel

DEFAULT_MODEL_DIR = Path(
    "/scratch/s5f/wedigs.s5f/training_runs/10-2026_04_28_09_24_50-tmc_0"
)
DATASET_DIR = Path("/scratch/s5f/wedigs.s5f/datasets/tmqm")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    output_dir = model_dir / "analysis"

    dataset = MoleculeDataset.open_existing_dataset_from_dir(DATASET_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors_path = output_dir / "descriptors.pt"

    if descriptors_path.exists():
        descriptors = torch.load(descriptors_path)
    else:
        model = load_model(model_dir)
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
            # DescriptorDistributionTask(),
            # TopNormDescriptorsTask(n_top=10),
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
            # # Single-file benchmark scorecard — the headline cross-model
            # # comparable artifact. See the docstring of
            # # DescriptorStructureBenchmarkTask for the protocol.
            # DescriptorStructureBenchmarkTask(),
            # # Companion variant: cluster on raw (unnormalized) descriptors with
            # # euclidean UMAP so the vector norm enters the partition. Lets you
            # # ask whether descriptor magnitude carries chemical signal that the
            # # canonical cosine path discards.
            # DescriptorStructureBenchmarkTask(
            #     cluster_metric="euclidean",
            #     use_raw_descriptors=True,
            #     summary_file_name="descriptor_structure_benchmark_euclidean_raw.yaml",
            # ),
            # *(
            #     HDBSCANClusterTask(
            #         min_cluster_size=mcs,
            #         figure_file_name=f"hdbscan_clusters_mcs{mcs}.png",
            #         summary_file_name=f"hdbscan_clusters_mcs{mcs}.yaml",
            #     )
            #     for mcs in (50, 200, 1000, 5000)
            # ),
            # # Coarse mcs collapses to one mega-cluster; the chemically coherent
            # # structure lives at small mcs. Sweep to find the purity-optimal
            # # granularity, then fingerprint each candidate. Run both with the
            # # clustering-UMAP and directly on the 64-D descriptors (no UMAP) to
            # # tell whether weak chemical clustering is intrinsic to the
            # # descriptor or introduced by the UMAP compression.
            # *(
            #     ClusterGranularitySweepTask(
            #         cluster_reducer=reducer,
            #         min_cluster_sizes=[25, 50, 100],
            #         summary_file_name=f"cluster_granularity_sweep_{tag}.yaml",
            #     )
            #     for reducer, tag in (("umap", "umap"), ("none", "noumap"))
            # ),
            # *(
            #     ClusterChemicalFingerprintTask(
            #         cluster_reducer=reducer,
            #         min_cluster_size=mcs,
            #         summary_file_name=f"cluster_fingerprint_{tag}_mcs{mcs}.yaml",
            #     )
            #     for reducer, tag in (("umap", "umap"), ("none", "noumap"))
            #     for mcs in (25, 50, 100)
            # ),
            # # For each clustering granularity, ask which chemical axis (metal
            # # block, geometry, ligand motif, donor element, …) best explains
            # # that partition — not just whether donor-set works.
            # *(
            #     ClusterAxisAnalysisTask(
            #         cluster_reducer=reducer,
            #         min_cluster_size=mcs,
            #         summary_file_name=f"cluster_axis_{tag}_mcs{mcs}.yaml",
            #     )
            #     for reducer, tag in (("umap", "umap"), ("none", "noumap"))
            #     for mcs in (25, 50, 100)
            # ),
            # # One viewer to manually inspect cluster chemistry on the settled
            # # UMAP: the small granularities switchable; heavy per-element /
            # # formula columns dropped to keep the JSON viewer-loadable.
            # ChemiscopeClusterTask(
            #     min_cluster_sizes=[25, 50, 100],
            #     default_color_mcs=50,
            #     include_heavy_properties=False,
            #     file_name="chemiscope_clusters.json.gz",
            # ),
        ],
    )

    # Serialize each task as it finishes so a timeout still leaves the
    # completed reports on disk for inspection.
    runner.run(descriptors=descriptors, dataset=dataset, output_dir=output_dir)


if __name__ == "__main__":
    main()
