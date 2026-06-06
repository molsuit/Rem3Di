"""Build the Chiro small-enantiomer docking-ranking dataset from its pickles.

The Chiro dataset (arXiv:2110.04383) ships three pandas ``.pkl`` DataFrames --
train / validation / test -- of RDKit Mols each carrying one 3D conformer plus a
molecule-level docking ``top_score``. Geometries are supplied directly, so this
follows the QM9 / chiral_cat ingest path: ``FilterAtomsStage`` + ``CopyDataStage``
with no conformer generation. Explicit hydrogens are re-placed from the
heavy-atom geometry inside the generator (``AddHs(addCoords=True)``) so MACE-OFF
sees complete molecules; the chiral tag on each Mol is dropped when it collapses
to coordinates, giving the honest chirality-from-geometry benchmark.

The single regression column is the per-conformer ``top_score``; the headline
metric is pairwise enantiomer ranking (see ``chiral_docking`` in the registry and
``evaluation/benchmark/pairwise.py``). The literature train/valid/test split is
read from the separate files and materialized into the zarr ``split`` column. A
``benchmark_manifest.yaml`` sidecar is written so the eval framework auto-
discovers the zarr as the ``chiral_docking`` benchmark.

Run with:
    uv run python scripts/dataset_creation/build_chiro_docking.py
    uv run python scripts/dataset_creation/build_chiro_docking.py \\
        --raw-root /p/scratch/mace/wedig1/raw_datasets/chiral_chiro_datasets \\
        --output /p/scratch/mace/wedig1/datasets/chiral_docking \\
        --max-conformers 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from threedscriptors.data_handling.benchmarks import (
    BenchmarkManifest,
    ChiroDockingBenchmark,
    get_benchmark,
)
from threedscriptors.data_handling.dataset.tasks import ElementSet, Split
from threedscriptors.data_handling.dataset_creation.generators.chiro_docking_generator import (
    ChiroDockingGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = Path("/p/scratch/mace/wedig1/raw_datasets/chiral_chiro_datasets")
DEFAULT_OUTPUT = Path("/p/scratch/mace/wedig1/datasets/chiral_docking")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--element-set",
        type=ElementSet,
        default=ElementSet.mace_off,
        choices=list(ElementSet),
        help="Atom-gate element preset (mace_off = drug subset H,C,N,O,F,P,S,Cl,Br,I).",
    )
    parser.add_argument(
        "--max-conformers",
        type=int,
        default=None,
        help=(
            "Conformers ingested per stereoisomer (first-N in file order). "
            "Defaults to the registry value for chiral_docking."
        ),
    )
    args = parser.parse_args()

    benchmark = get_benchmark("chiral_docking")
    assert isinstance(benchmark, ChiroDockingBenchmark)

    split_files = {
        Split.train: args.raw_root / benchmark.train_file,
        Split.valid: args.raw_root / benchmark.valid_file,
        Split.test: args.raw_root / benchmark.test_file,
    }
    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Chiro docking {split.name} pickle not found: {path}")

    if args.output.exists():
        logger.info("%s exists; delete the dir to rebuild. Skipping.", args.output)
        return

    cap = (
        args.max_conformers
        if args.max_conformers is not None
        else benchmark.max_conformers_per_stereoisomer
    )
    generator = ChiroDockingGenerator(
        split_files=split_files,
        max_conformers_per_stereoisomer=cap,
        id_column=benchmark.id_column,
        nonstereo_column=benchmark.nonstereo_column,
        mol_column=benchmark.mol_column,
        score_column=benchmark.score_column,
    )
    filter_stage = FilterAtomsStage(
        config=FilterAtomsStageConfig(
            element_set=args.element_set,
            # Guard against degenerate geometries (overlapping atoms) that make
            # MACE emit NaN embeddings. 0.5 A is well below any real bond length.
            min_interatomic_distance=0.5,
        )
    )
    copy_data = CopyDataStage(dtype=torch.float64)

    creation_config = DatasetCreationConfig(path=args.output, N_structures=None)
    dataset_config = DatasetConfig(tasks=generator.task_set())

    DatasetConstructionOrchestrator(
        pipeline=[filter_stage, copy_data],
        batch_generator=generator,
        construction_config=creation_config,
        dataset_config=dataset_config,
    ).build_dataset()

    BenchmarkManifest.from_benchmark(benchmark).to_zarr_dir(args.output)
    logger.info(
        "chiral_docking: generator raw=%d kept=%d invalid=%d conf-capped=%d (cap=%s) | "
        "FilterAtomsStage kept=%d dropped=%d (%s) -> %s",
        generator.load_stats.n_raw_rows,
        generator.load_stats.n_kept,
        generator.load_stats.n_invalid_smiles,
        generator.load_stats.n_filtered_out,
        cap,
        filter_stage.load_stats.n_kept,
        filter_stage.load_stats.n_filtered_out,
        args.element_set.value,
        args.output,
    )


if __name__ == "__main__":
    main()
