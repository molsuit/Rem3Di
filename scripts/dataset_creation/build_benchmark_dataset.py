"""Build the full benchmark panel (MoleculeNet + TDC + Polaris).

Single yaml-driven entry point. The script dispatches on ``benchmark.source``
to the matching generator and writes one zarr per benchmark under
``output_root/<dataset_id>/``, with the literature split materialized in the
zarr ``split`` column. Datasets whose zarr already exists are skipped, so
re-running is idempotent and rebuilding one benchmark just means deleting its
subdirectory.

Polaris benchmarks are only built when ``polaris_raw_root`` is set in the
config and the per-dataset parquet has been produced by
``scripts/dataset_download/dump_polaris.py`` (which runs in its own ephemeral
venv -- polaris-lib pins ``zarr<3`` and cannot share this venv). Polaris
benchmarks whose parquet is missing are skipped with a warning.

Usage::

    uv run python scripts/dataset_creation/build_benchmark_dataset.py \\
        --config configs/dataset_creation/benchmarks_local.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml
import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.benchmarks import (
    MOLECULENET_BENCHMARKS,
    POLARIS_BENCHMARKS,
    TDC_BENCHMARKS,
    BenchmarkManifest,
    MoleculeNetBenchmark,
    PolarisBenchmark,
    TdcBenchmark,
)
from threedscriptors.data_handling.dataset_creation.build_config import (
    BenchmarkBuildConfig,
)
from threedscriptors.data_handling.dataset_creation.generators.moleculenet_generator import (
    MoleculeNetGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.polaris_offline_generator import (
    PolarisOfflineGenerator,
)
from threedscriptors.data_handling.dataset_creation.generators.tdc_generator import (
    TdcGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    ConformerGenerationStage,
    CopyDataStage,
    FilterMoleculeStage,
)

logger = logging.getLogger(__name__)


def _make_generator(
    benchmark: MoleculeNetBenchmark | TdcBenchmark | PolarisBenchmark,
    cfg: BenchmarkBuildConfig,
):
    if isinstance(benchmark, MoleculeNetBenchmark):
        # MoleculeNet does filter + scaffold-split together (see generator
        # docstring) so it doesn't route through FilterMoleculeStage. It
        # still shares the filter knobs via FilterMoleculeStageConfig.
        return MoleculeNetGenerator(
            benchmark,
            cfg.moleculenet_raw_root,
            batch_size=cfg.batch_size,
            filter_config=cfg.filter,
        )
    if isinstance(benchmark, PolarisBenchmark):
        if cfg.polaris_raw_root is None:
            raise ValueError(
                "polaris_raw_root is unset; cannot build polaris benchmarks. "
                "Dump the parquet first via scripts/dataset_download/dump_polaris.py."
            )
        return PolarisOfflineGenerator(
            benchmark,
            cfg.polaris_raw_root,
            batch_size=cfg.batch_size,
        )
    return TdcGenerator(
        benchmark,
        cfg.tdc_cache,
        batch_size=cfg.batch_size,
        seed=cfg.tdc_train_valid_seed,
    )


def build_one(
    benchmark: MoleculeNetBenchmark | TdcBenchmark | PolarisBenchmark,
    cfg: BenchmarkBuildConfig,
) -> None:
    zarr_path = cfg.output_root / benchmark.dataset_id
    if zarr_path.exists():
        logger.info(
            "%s: %s exists; skipping (delete the dir to rebuild)",
            benchmark.dataset_id,
            zarr_path,
        )
        return

    # Polaris: skip with a warning if the parquet hasn't been dumped yet so a
    # partial polaris setup doesn't block the rest of the panel from building.
    if isinstance(benchmark, PolarisBenchmark):
        if cfg.polaris_raw_root is None:
            logger.warning(
                "%s: polaris_raw_root unset; skipping. "
                "Run scripts/dataset_download/dump_polaris.py and set "
                "polaris_raw_root in the build config to enable.",
                benchmark.dataset_id,
            )
            return
        parquet_path = cfg.polaris_raw_root / benchmark.parquet_filename
        if not parquet_path.exists():
            logger.warning(
                "%s: parquet %s missing; skipping. Dump it via "
                "scripts/dataset_download/dump_polaris.py --slug %s --out %s",
                benchmark.dataset_id,
                parquet_path,
                benchmark.polaris_slug,
                parquet_path,
            )
            return

    creation_config = DatasetCreationConfig(
        path=zarr_path,
        max_embed_attempts=cfg.max_embed_attempts,
        max_MMFF_steps=cfg.max_mmff_steps,
        mmff_non_bonded_thresh=cfg.mmff_non_bonded_thresh,
        N_sampled_conformers=cfg.n_sampled_conformers,
    )
    dataset_config = DatasetConfig(
        atom_chunk=cfg.atom_chunk,
        molecule_chunk=cfg.molecule_chunk,
        atom_chunks_per_shard=cfg.atom_chunks_per_shard,
        molecule_chunks_per_shard=cfg.molecule_chunks_per_shard,
        contains_smiles=cfg.contains_smiles,
        tasks=benchmark.task_set(),
    )
    generator = _make_generator(benchmark, cfg)
    pipeline = [
        ConformerGenerationStage(dataset_creation_config=creation_config),
        CopyDataStage(dtype=torch.float64),
    ]
    # TDC and Polaris emit raw SMILES and route through FilterMoleculeStage
    # (parse / canonicalize / dedupe / filter). MoleculeNet filters inside the
    # generator alongside scaffold-split, since the split is a global operation
    # over the kept-SMILES set; ``apply_smiles_filter`` keeps the rules shared.
    if isinstance(benchmark, (TdcBenchmark, PolarisBenchmark)):
        pipeline.insert(0, FilterMoleculeStage(config=cfg.filter))

    logger.info("%s: building -> %s", benchmark.dataset_id, zarr_path)
    DatasetConstructionOrchestrator(
        pipeline=pipeline,
        batch_generator=generator,
        construction_config=creation_config,
        dataset_config=dataset_config,
    ).build_dataset()
    # Sidecar manifest: eval reads this + dataset_config.yaml + the zarr; it
    # never imports the registry.
    BenchmarkManifest.from_benchmark(benchmark).to_zarr_dir(zarr_path)
    logger.info("%s: done", benchmark.dataset_id)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = pyd_yaml.parse_yaml_file_as(BenchmarkBuildConfig, args.config)
    selected = cfg.only_datasets
    for bench in (*MOLECULENET_BENCHMARKS, *TDC_BENCHMARKS, *POLARIS_BENCHMARKS):
        if selected is not None and bench.dataset_id not in selected:
            continue
        build_one(bench, cfg)


if __name__ == "__main__":
    main()
