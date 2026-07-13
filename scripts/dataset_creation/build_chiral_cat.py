"""Build the ChiralCat chirality-type classification dataset from its extxyz.

ChiralCat ships 3D structures directly (one ETKDG/MMFF94 conformer per molecule,
explicit H, label + SMILES on each comment line), so this follows the QM9 ingest
path: ``FilterAtomsStage`` + ``CopyDataStage`` with no conformer generation. The
single 5-class chirality label is written as a ``TaskType.multiclass`` system
task and a group-aware, class-stratified train/valid/test split is materialized
into the zarr ``split`` column (see ``ChiralCatGenerator``).

The element set is ``mace_polar`` (Z 1-83) so the ~100 organometallic
``[Fe]``/CO fragments noted in the dataset card are kept — only the ``mace_off``
drug subset would drop them. A ``benchmark_manifest.yaml`` sidecar is written so
the eval framework auto-discovers the zarr as the ``chiral_cat`` benchmark.

Run with:
    uv run python scripts/dataset_creation/build_chiral_cat.py
    uv run python scripts/dataset_creation/build_chiral_cat.py \\
        --raw-root /p/scratch/mace/wedig1/raw_datasets/chiral_cat \\
        --output /p/scratch/mace/wedig1/datasets/chiral_cat
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from remedi.data_handling.benchmarks import (
    BenchmarkManifest,
    LocalXyzBenchmark,
    get_benchmark,
)
from remedi.data_handling.dataset.tasks import ElementSet
from remedi.data_handling.dataset_creation.generators.chiral_cat_generator import (
    ChiralCatGenerator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = Path("/p/scratch/mace/wedig1/raw_datasets/chiral_cat")
DEFAULT_OUTPUT = Path("/p/scratch/mace/wedig1/datasets/chiral_cat")


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
        default=ElementSet.mace_polar,
        choices=list(ElementSet),
        help="Atom-gate element preset (mace_polar keeps organometallics).",
    )
    args = parser.parse_args()

    benchmark = get_benchmark("chiral_cat")
    assert isinstance(benchmark, LocalXyzBenchmark)
    xyz_file = args.raw_root / benchmark.xyz_filename
    if not xyz_file.exists():
        raise FileNotFoundError(f"ChiralCat extxyz not found: {xyz_file}")

    if args.output.exists():
        logger.info("%s exists; delete the dir to rebuild. Skipping.", args.output)
        return

    generator = ChiralCatGenerator(
        xyz_file=xyz_file,
        label_key=benchmark.label_key,
        smiles_key=benchmark.smiles_key,
    )
    filter_stage = FilterAtomsStage(
        config=FilterAtomsStageConfig(
            element_set=args.element_set,
            # Drop degenerate geometries (overlapping atoms) — ~42 ChiralCat
            # conformers have atoms at identical coordinates, which makes MACE
            # emit NaN embeddings. 0.5 A is well below any real bond length.
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
    # Surface both gates explicitly: the generator's SMILES gate and the
    # atom-side element/heavy-atom gate (e.g. a bare [H+] proton has no heavy
    # atom and is dropped here). Neither should be silent.
    logger.info(
        "chiral_cat: generator raw=%d kept=%d invalid_smiles=%d | "
        "FilterAtomsStage kept=%d dropped=%d (%s) -> %s",
        generator.load_stats.n_raw_rows,
        generator.load_stats.n_kept,
        generator.load_stats.n_invalid_smiles,
        filter_stage.load_stats.n_kept,
        filter_stage.load_stats.n_filtered_out,
        args.element_set.value,
        args.output,
    )


if __name__ == "__main__":
    main()
