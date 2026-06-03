"""Run the vector-retrieval eval.

Two ways to invoke:

1. From a yaml config (declarative, preferred for real runs)::

    uv run python scripts/evaluation/run_retrieval_eval.py \\
        --config configs/eval/retrieval_pcqm100k.yaml

2. Directly against a model checkpoint + dataset, building the config in-process
   (handy for the pcqm_ablation / pcqm100k smoke test)::

    uv run python scripts/evaluation/run_retrieval_eval.py \\
        --model-dir /p/scratch/mace/wedig1/training_runs/pcqm_ablation/24-2026_05_19_12_57_28-pcqm_baseline \\
        --dataset /p/scratch/mace/wedig1/datasets/pcqm4m/pcqm100k \\
        --output-dir reports/retrieval/pcqm100k_baseline

``--model-dir`` must point at a *single* trained run directory (the one holding
``post_training_architecture_config.yaml`` + ``encoder.pth`` + the two
preprocessor ``.pth`` files), not the ablation parent directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydantic_yaml as pyd_yaml

from threedscriptors.evaluation.benchmark.descriptors import RemediConfig
from threedscriptors.evaluation.retrieval.config import (
    NearestMoleculeTaskConfig,
    RetrievalEvalConfig,
    TanimotoSimilarityTaskConfig,
)
from threedscriptors.evaluation.retrieval.runner import run_retrieval_eval

logger = logging.getLogger(__name__)


def _config_from_args(args: argparse.Namespace) -> RetrievalEvalConfig:
    model_dir = Path(args.model_dir)
    model_name = args.model_name or model_dir.name
    return RetrievalEvalConfig(
        dataset_path=Path(args.dataset),
        dataset_id=args.dataset_id,
        model=RemediConfig(
            name=model_name,
            model_dir=model_dir,
            batch_size=args.batch_size,
            device=args.device,
            mace_model_path=args.mace_model_path,
        ),
        output_dir=Path(args.output_dir),
        max_structures=args.max_structures,
        tasks=[
            TanimotoSimilarityTaskConfig(
                k=args.k,
                n_query_sample=args.n_query_sample,
                n_global_pairs=args.n_global_pairs,
            ),
            NearestMoleculeTaskConfig(
                k=args.k,
                query_smiles=list(args.query_smiles or []),
                n_random_query_smiles=args.n_query_smiles,
                query_sample_seed=args.query_sample_seed,
            ),
        ],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--mace-model-path",
        type=Path,
        default=None,
        help="Override the MACE foundation-model path baked into the checkpoint "
        "config (needed when running on a different machine than training).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/p/scratch/mace/wedig1/datasets/pcqm4m/pcqm100k"),
    )
    parser.add_argument("--dataset-id", type=str, default="pcqm100k")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/retrieval"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-query-sample", type=int, default=500)
    parser.add_argument("--n-global-pairs", type=int, default=50_000)
    parser.add_argument("--max-structures", type=int, default=None)
    parser.add_argument(
        "--n-query-smiles",
        type=int,
        default=10,
        help="Number of query SMILES to sample at random from the dataset for "
        "the nearest-molecule task.",
    )
    parser.add_argument("--query-sample-seed", type=int, default=0)
    parser.add_argument(
        "--query-smiles",
        nargs="*",
        default=None,
        help="Explicit extra query SMILES (added on top of the random sample).",
    )
    args = parser.parse_args()

    if args.config is not None:
        cfg = pyd_yaml.parse_yaml_file_as(RetrievalEvalConfig, args.config)
    elif args.model_dir is not None:
        cfg = _config_from_args(args)
    else:
        parser.error("provide either --config or --model-dir")

    report = run_retrieval_eval(cfg)
    logger.info(
        "Retrieval eval done: %d structures, dim=%d, %d Tanimoto + %d nearest tasks",
        report.n_structures,
        report.embedding_dim,
        len(report.tanimoto_results),
        len(report.nearest_molecule_results),
    )


if __name__ == "__main__":
    main()
