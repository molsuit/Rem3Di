"""Capacity / distribution diagnostic for the PCQM ablation models.

Trims the tmQM clustering pipeline to its descriptor-intrinsic tasks (no
projection, no clustering, no chemiscope) and pre-subsamples the 3.5M-molecule
PCQM set so the encoder forward pass is the only step that scales with the
dataset. The capacity diagnostic's stats (per-dim marginal entropy, effective
dimension, explained-variance spectrum) saturate well below 3.5M samples, so a
200k subsample is the working budget.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

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
    DescriptorAnalysisRunner,
    DescriptorDistributionTask,
    DescriptorNormalizationConfig,
    TopNormDescriptorsTask,
)
from remedi.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from remedi.model.remedi_model import REM3DIModel

DEFAULT_MODEL_DIR = Path(
    "/path/to/training_runs/pcqm_ablation/" "24-2026_05_19_12_57_28-pcqm_baseline"
)
DEFAULT_DATASET_DIR = Path("/path/to/datasets/pcqm4m/pcqm_only_structures_3_5_M")
DEFAULT_N_SAMPLES = 200_000
DEFAULT_SEED = 0


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


def _draw_sample_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_samples, replace=False)
    # Sort so zarr shard reads in the encoder pass walk the file sequentially
    # — random-access reads through sharded zarr v3 are noticeably slower.
    return np.sort(indices).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of molecules to subsample before encoding (default 200k).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for the encoder pass.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Per-worker prefetch (ignored when --num-workers=0).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="capacity_analysis",
        help="Output sub-directory inside model_dir.",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    output_dir = model_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MoleculeDataset.open_existing_dataset_from_dir(args.dataset_dir)
    n_total = int(dataset.N_structures)
    n_samples = min(args.n_samples, n_total)

    descriptors_path = output_dir / "descriptors.pt"
    indices_path = output_dir / "sample_indices.npy"

    if descriptors_path.exists() and indices_path.exists():
        descriptors = torch.load(descriptors_path)
        sample_indices = np.load(indices_path)
    else:
        sample_indices = _draw_sample_indices(n_total, n_samples, args.seed)
        full_train_dataset = TrainingMoleculeDataset.from_molecule_dataset(
            dataset, get_item=atoms_getitem
        )
        subset = Subset(full_train_dataset, sample_indices.tolist())

        model = load_model(model_dir)
        descriptors = evaluate_molecular_descriptor_on_dataset(
            model,
            subset,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        )
        torch.save(descriptors, descriptors_path)
        np.save(indices_path, sample_indices)

    runner = DescriptorAnalysisRunner(
        normalization=DescriptorNormalizationConfig(z_score=False, l2_normalize=True),
        # No 2D projection: every downstream task here is descriptor-intrinsic,
        # so we skip the UMAP fit (impractical at this scale).
        projection=None,
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
        ],
    )

    runner.run(
        descriptors=descriptors,
        dataset=dataset,
        output_dir=output_dir,
        sample_indices=sample_indices,
    )


if __name__ == "__main__":
    main()
