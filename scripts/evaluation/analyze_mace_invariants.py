"""Compute streaming statistics + information-theoretic metrics over the
atomwise MACE invariant descriptors that the AtomicDescriptorPreprocessor
operates on.

Outputs (per output_dir):
    mace_invariant_report.yaml
    mean_std_per_dim.png
    l2_norm_distribution.png
    per_dim_entropy.png
    explained_variance.png

The script intentionally avoids the AtomicDescriptorPreprocessor's
normalization layer: we want to see what the MACE backbone produces *before*
the preprocessor recentres it, since that's what the preprocessor stats are
fit on. (If you want the post-normalization view, point ``mace_config`` at the
same checkpoint and rerun after loading the trained ``atomic_preprocessor.pth``
on top — the script structure is unchanged.)

Usage:
    uv run python scripts/evaluation/analyze_mace_invariants.py \\
        --config scripts/evaluation/analyze_mace_invariants.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pydantic_yaml as pyaml
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from threedscriptors.configuration.mace_config import MaceConfig
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from threedscriptors.data_handling.sample import Sample, yield_molecules_collate_fn
from threedscriptors.evaluation.descriptor_analysis.mace_invariant_stats import (
    AnalysisReport,
    LayerSpec,
    LayerStreamStats,
    get_layer_invariant_specs,
    plot_explained_variance,
    plot_l2_norm_per_layer,
    plot_mean_std_per_dim,
    plot_per_dim_entropy,
    summarise_layer,
    summarise_total,
)
from threedscriptors.model.preprocessing.preprocessing import _sample_to_simstate
from threedscriptors.training.data import worker_init_fn
from threedscriptors.utils.model_utils import get_invariant_indices

logger = logging.getLogger("mace_invariant_analysis")


class MaceInvariantAnalysisConfig(BaseModel):
    """YAML-driven configuration for the analysis run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mace_config: MaceConfig
    dataset_path: Path
    output_dir: Path

    batch_size: int = 16
    num_workers: int = 4
    max_batches: int | None = None
    seed: int = 0

    histogram_bins: int = 128
    coding_rate_eps: float = 0.5
    dead_threshold: float = 0.2
    reservoir_capacity: int = 50_000


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "analysis.log", mode="w")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _run_mace_on_sample(mace_model, sample: Sample) -> torch.Tensor:
    """Build a SimState from a flat-batched Sample and return MACE node feats.

    Delegates to the official mace-torch wrapper (MaceTorchSimModel) so that
    PolarMACE's data_dict (rcell, volume, fermi_level, external_field,
    density_coefficients) is constructed by the same code path the rest of
    this repo uses.
    """
    from threedscriptors.model.preprocessing.preprocessing import _capture_node_feats

    state, _ = _sample_to_simstate(sample, r_max=float(mace_model.r_max))
    return _capture_node_feats(mace_model, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a MaceInvariantAnalysisConfig YAML.",
    )
    args = parser.parse_args()

    cfg = pyaml.parse_yaml_file_as(MaceInvariantAnalysisConfig, args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(cfg.output_dir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.mace_config.device)
    logger.info("Loading MACE model from %s", cfg.mace_config.model_path)
    mace_model = cfg.mace_config.build_torch_sim_model()
    mace_model.eval()

    input_irreps = cfg.mace_config.get_irrep_signature()
    invariant_indices, invariant_irreps = get_invariant_indices(input_irreps)
    layer_specs = get_layer_invariant_specs(input_irreps)
    invariant_dim = invariant_irreps.dim
    logger.info("Input irreps: %s", input_irreps)
    logger.info("Invariant irreps: %s (dim=%d)", invariant_irreps, invariant_dim)
    for spec in layer_specs:
        logger.info(
            "  layer %d: invariant indices [%d, %d) (width=%d)",
            spec.layer_index,
            spec.start,
            spec.stop,
            spec.width,
        )

    invariant_idx_t = torch.tensor(invariant_indices, dtype=torch.long, device=device)

    # Persist the analysis config alongside outputs for reproducibility.
    pyaml.to_yaml_file(cfg.output_dir / "analysis_config.yaml", cfg)

    logger.info("Opening dataset at %s", cfg.dataset_path)
    dataset = TrainingMoleculeDataset(
        cfg.dataset_path, get_item=atoms_getitem, in_memory=True
    )
    logger.info("Dataset size: %d molecules", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=cfg.num_workers,
        shuffle=True,
        collate_fn=yield_molecules_collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    layer_stats = [
        LayerStreamStats(
            spec=spec,
            device=device,
            reservoir_capacity=cfg.reservoir_capacity,
            seed=cfg.seed,
        )
        for spec in layer_specs
    ]
    total_stats = LayerStreamStats(
        spec=LayerSpec(layer_index=-1, start=0, stop=invariant_dim),
        device=device,
        reservoir_capacity=cfg.reservoir_capacity,
        seed=cfg.seed,
    )

    n_atoms_seen = 0
    max_batches = cfg.max_batches if cfg.max_batches is not None else len(loader)

    with torch.inference_mode():
        for batch_idx, sample in enumerate(loader):
            if batch_idx >= max_batches:
                break
            sample.to_(device)
            node_feats = _run_mace_on_sample(mace_model, sample).detach()
            invariants = node_feats.index_select(-1, invariant_idx_t)
            n_atoms_seen += invariants.shape[0]

            for layer_stat in layer_stats:
                spec = layer_stat.spec
                layer_stat.update(invariants[:, spec.start : spec.stop])
            total_stats.update(invariants)

            if (batch_idx + 1) % 25 == 0:
                logger.info(
                    "batch %d/%d  atoms=%d  cur_total_l2_mean=%.3f",
                    batch_idx + 1,
                    max_batches,
                    n_atoms_seen,
                    float(total_stats.l2_norms[-1].mean()),
                )

    logger.info("Streaming pass complete. atoms=%d", n_atoms_seen)

    layer_summaries = [
        summarise_layer(
            s,
            histogram_bins=cfg.histogram_bins,
            coding_rate_eps=cfg.coding_rate_eps,
            dead_threshold=cfg.dead_threshold,
        )
        for s in layer_stats
    ]
    total_summary = summarise_total(total_stats, coding_rate_eps=cfg.coding_rate_eps)

    report = AnalysisReport(
        input_irreps=str(input_irreps),
        invariant_irreps=str(invariant_irreps),
        layers=layer_summaries,
        total=total_summary,
        histogram_bins=cfg.histogram_bins,
    )
    pyaml.to_yaml_file(cfg.output_dir / "mace_invariant_report.yaml", report)

    fig = plot_mean_std_per_dim(report)
    fig.savefig(cfg.output_dir / "mean_std_per_dim.png", dpi=150)

    total_norms = np.concatenate(total_stats.l2_norms)
    fig = plot_l2_norm_per_layer(layer_stats, total_norms)
    fig.savefig(cfg.output_dir / "l2_norm_distribution.png", dpi=150)

    fig = plot_per_dim_entropy(report, dead_threshold=cfg.dead_threshold)
    fig.savefig(cfg.output_dir / "per_dim_entropy.png", dpi=150)

    fig = plot_explained_variance(report)
    fig.savefig(cfg.output_dir / "explained_variance.png", dpi=150)

    logger.info("=== Summary ===")
    for s in layer_summaries:
        logger.info(
            "Layer %d (width=%d, n=%d): "
            "L2 mean=%.3f std=%.3f  PR=%.2f  R(eps=%.2f)=%.2f bits  "
            "mean_abs_corr=%.3f  H_tot=%.1f  dead=%d",
            s.layer_index,
            s.width,
            s.n_atoms,
            s.l2_norm_mean,
            s.l2_norm_std,
            s.participation_ratio,
            s.coding_rate_eps,
            s.coding_rate,
            s.mean_abs_correlation,
            s.H_total,
            s.dead_dims,
        )
    logger.info(
        "Total invariants (dim=%d): "
        "L2 mean=%.3f std=%.3f  PR=%.2f  R(eps=%.2f)=%.2f bits  mean_abs_corr=%.3f",
        total_summary.invariant_dim,
        total_summary.l2_norm_mean,
        total_summary.l2_norm_std,
        total_summary.participation_ratio,
        total_summary.coding_rate_eps,
        total_summary.coding_rate,
        total_summary.mean_abs_correlation,
    )

    logger.info("Wrote outputs to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
