import argparse
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pydantic_yaml as pyaml
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, Subset

from remedi.configuration.architecture_config import (
    ArchitectureConfig,
)
from remedi.configuration.training_config import PretrainingConfig, ProbeConfig
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, save_checkpoint
from remedi.training.data.mixture import MixtureDatasets, resolve_mixture
from remedi.training.data.sample import yield_molecules_collate_fn
from remedi.training.data.training_dataset import atoms_getitem
from remedi.training.linear_probe import LinearProbeMonitor
from remedi.training.pretraining import denoising_step, vicreg_descriptor_loss
from remedi.training.telemetry import TrainingTelemetry

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch._dynamo  # noqa: E402 - after the device and logging setup

torch._logging.set_logs(recompiles=True)

# nvalchemiops applies @torch.compile to prepare_batch_idx_ptr, which is on the
# MACE neighbor-list path called every step. Bucket sampling makes the input
# shapes (total atoms, num molecules) vary, triggering repeated recompiles. The
# function does ~µs of cumsum/bincount work — not worth compiling. Override the
# decorated symbol with a dynamo-disabled version before any model is built.
import nvalchemiops.torch.neighbors.neighbor_utils as _nv_neighbor_utils  # noqa: E402 - patched before any model is built

_nv_neighbor_utils.prepare_batch_idx_ptr = torch._dynamo.disable(
    _nv_neighbor_utils.prepare_batch_idx_ptr
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run online embedding denoising pretraining."
    )
    parser.add_argument(
        "--training_config",
        type=Path,
        required=True,
        help="Path to the training_config.yaml file.",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Optional cap on the number of training structures for smoke runs.",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=None,
        help="Optional cap on the validation structures per component for smoke runs.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Optional epoch override for smoke runs.",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional cap on training batches per epoch for smoke runs.",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per epoch for smoke runs.",
    )
    return parser.parse_args()


def build_probe_monitor(
    probe_config: ProbeConfig,
    datasets: MixtureDatasets,
    collate_fn,
    device,
    logger,
) -> LinearProbeMonitor | None:
    """Build the frozen linear-probe monitor on labelled validation structures.

    Returns None (with a warning) when probing is disabled or the first mixture
    component has no labels to probe.
    """
    if not probe_config.enabled:
        return None
    probe_set = datasets.probe_set(
        structures=probe_config.n_probe_molecules,
        seed=probe_config.seed,
        properties=probe_config.properties,
    )
    if probe_set is None:
        logger.warning(
            "Probe enabled but the first mixture component has no labels to probe "
            "(or none of %s); skipping.",
            probe_config.properties,
        )
        return None
    count = len(probe_set.dataset)
    if count < 4:
        logger.warning("Validation split too small for probe (%d); skipping.", count)
        return None
    logger.info(
        "Linear probe enabled: M=%d cols=%s every_n_steps=%d",
        count,
        probe_set.names,
        probe_config.every_n_steps,
    )
    return LinearProbeMonitor(
        probe_samples=[probe_set.dataset[i] for i in range(count)],
        labels=probe_set.labels,
        masks=probe_set.masks,
        property_names=probe_set.names,
        collate_fn=collate_fn,
        device=device,
        every_n_steps=probe_config.every_n_steps,
        ridge_alpha=probe_config.ridge_alpha,
        val_fraction=probe_config.val_fraction,
        batch_size=min(256, count),
        report_count_accuracy=probe_config.report_count_accuracy,
        seed=probe_config.seed,
    )


def first(dataset: Dataset, lengths: np.ndarray, count: int | None):
    """The first ``count`` structures (all when ``count`` is None)."""
    if count is None or count >= len(lengths):
        return dataset, lengths
    return Subset(dataset, range(count)), lengths[:count]


def setup_logging(filename, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("remedi")
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.FileHandler(filename=filename, mode="w")
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
        logger.propagate = False
    return logger


def main():  # noqa: C901 - one training loop
    args = parse_args()

    training_config = pyaml.parse_yaml_file_as(PretrainingConfig, args.training_config)
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    if args.max_epochs is not None:
        training_config.epochs = args.max_epochs
    architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig, training_config.model_config_path
    )

    training_data_dir = training_config.create_run_directory()

    logger = setup_logging(filename=training_data_dir / "logfile.info")
    logger.info(
        "Smoke caps: max_train=%s max_val=%s max_epochs=%s "
        "max_train_batches=%s max_val_batches=%s",
        args.max_train,
        args.max_val,
        args.max_epochs,
        args.max_train_batches,
        args.max_val_batches,
    )

    data = resolve_mixture(training_config.data)
    data.write_record(training_data_dir)
    datasets = data.open_datasets(atoms_getitem)
    train_dataset, train_lengths = first(
        datasets.training, datasets.training_atom_counts, args.max_train
    )
    validation_sets = {
        name: first(dataset, lengths, args.max_val)
        for name, (dataset, lengths) in datasets.validation.items()
    }
    logger.info(
        "Mixture (%s): %d training structures; validation %s",
        training_config.data.domain,
        len(train_lengths),
        {name: len(lengths) for name, (_, lengths) in validation_sets.items()},
    )

    noise_level = training_config.noise_level
    training_loader = training_config.dataloader.build(
        train_dataset,
        lengths=train_lengths,
        collate_fn=yield_molecules_collate_fn,
        shuffle=True,
        seed=training_config.seed,
    )
    validation_loaders = {
        name: training_config.dataloader.build(
            dataset,
            lengths=lengths,
            collate_fn=yield_molecules_collate_fn,
            shuffle=False,
            seed=training_config.seed,
        )
        for name, (dataset, lengths) in validation_sets.items()
    }

    logger.info("Data Loaders Prepared")

    probe_monitor = build_probe_monitor(
        training_config.probe, datasets, yield_molecules_collate_fn, device, logger
    )

    bundle = architecture_config.build()
    preprocessor = bundle.preprocessor
    encoder = bundle.encoder
    decoder = bundle.decoder

    compile_cfg = training_config.compile
    torch._dynamo.config.cache_size_limit = compile_cfg.dynamo_cache_size_limit
    if hasattr(preprocessor, "pad_multiple"):
        preprocessor.pad_multiple = compile_cfg.pad_multiple

    all_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(preprocessor.parameters())
    )

    optimizer = torch.optim.AdamW(
        all_params,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[training_config.learning_rate],
        total_steps=training_config.epochs * len(training_loader),
    )

    pyaml.to_yaml_file(
        training_data_dir / ARCHITECTURE_FILENAME,
        architecture_config,
    )

    with TrainingTelemetry(
        wandb_active=training_config.wandb_active,
        run_name=training_config.training_name,
        group_name=training_config.run_group,
        out_dir=training_data_dir,
        config={
            "train_config": training_config.model_dump(),
            "architecture_config": architecture_config.model_dump(),
        },
    ) as telemetry:
        encoder.to(device, dtype=torch.float32)
        decoder.to(device, dtype=torch.float32)
        preprocessor.to(device)

        if compile_cfg.enabled:
            encoder = torch.compile(encoder, dynamic=True)
            decoder = torch.compile(decoder, dynamic=True)

        print("Training Start")

        vicreg_cfg = training_config.vicreg

        # Rolling throughput counters. Logged every WINDOW_STEPS batches so we
        # have a real atoms/sec number to A/B against, instead of squinting at
        # the wandb GPU-util plot. The first window includes compile warmup —
        # ignore it.
        WINDOW_STEPS = 50
        global_step = 0
        window_atoms = 0
        window_t0 = perf_counter()
        # Rolling VICReg accumulators, reset each throughput window so the
        # mid-epoch flush shows the var/cov *trajectory* rather than an
        # epoch-cumulative average that smooths over early collapse.
        window_var = torch.zeros((), device=device)
        window_cov = torch.zeros((), device=device)
        window_vc_steps = 0

        for epoch in range(training_config.epochs):
            # Initialize task and total train losses

            running = torch.zeros((), device=device)
            running_var = torch.zeros((), device=device)
            running_cov = torch.zeros((), device=device)
            running_zmax = torch.zeros((), device=device)

            encoder.train()
            decoder.train()
            preprocessor.train()

            optimizer.zero_grad()

            for batch_index, samples in enumerate(training_loader):
                if (
                    args.max_train_batches is not None
                    and batch_index >= args.max_train_batches
                ):
                    break
                samples.to_(device)
                step = denoising_step(
                    samples,
                    preprocessor=preprocessor,
                    encoder=encoder,
                    decoder=decoder,
                    noise_level=noise_level,
                )
                batch_atoms = step.atom_count
                molecular_descriptor = step.molecular_descriptor
                denoising_loss = step.denoising_loss

                if vicreg_cfg.enabled:
                    var_loss, cov_loss = vicreg_descriptor_loss(
                        molecular_descriptor.flat, target_std=vicreg_cfg.target_std
                    )
                    total_loss = (
                        denoising_loss
                        + vicreg_cfg.variance_weight * var_loss
                        + vicreg_cfg.covariance_weight * cov_loss
                    )
                    running_var += var_loss.detach()
                    running_cov += cov_loss.detach()
                    window_var += var_loss.detach()
                    window_cov += cov_loss.detach()
                    window_vc_steps += 1
                else:
                    total_loss = denoising_loss

                with torch.no_grad():
                    running_zmax = torch.maximum(
                        running_zmax,
                        molecular_descriptor.flat.norm(dim=-1).max().detach(),
                    )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    all_params, max_norm=training_config.max_grad_norm
                )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                running += denoising_loss.detach()

                window_atoms += batch_atoms
                global_step += 1
                if global_step % WINDOW_STEPS == 0:
                    dt = perf_counter() - window_t0
                    window_metrics: dict[str, float] = {
                        "atoms_per_s": window_atoms / dt,
                        "step_ms": 1000.0 * dt / WINDOW_STEPS,
                        "global_step": global_step,
                        "epoch_idx": epoch,
                    }
                    if vicreg_cfg.enabled and window_vc_steps > 0:
                        window_metrics["vicreg_variance_train_window"] = float(
                            (window_var / window_vc_steps).item()
                        )
                        window_metrics["vicreg_covariance_train_window"] = float(
                            (window_cov / window_vc_steps).item()
                        )
                    telemetry.log_metrics(window_metrics)
                    window_atoms = 0
                    window_t0 = perf_counter()
                    window_var = torch.zeros((), device=device)
                    window_cov = torch.zeros((), device=device)
                    window_vc_steps = 0

                if probe_monitor is not None:
                    probe_metrics = probe_monitor.maybe_run(
                        global_step, preprocessor, encoder
                    )
                    if probe_metrics:
                        telemetry.log_probe(global_step, probe_metrics)
                        logger.info(
                            "probe @ step %d: %s",
                            global_step,
                            {k: round(v, 4) for k, v in probe_metrics.items()},
                        )

            avg_train_loss = (
                running
                / (
                    (batch_index + 1)
                    * architecture_config.embedding_preprocess_config.output_irreps_dim
                )
            ).item()

            normaliser = (
                architecture_config.embedding_preprocess_config.output_irreps_dim
            )
            validation_loss_sum = torch.zeros((), device=device)
            validation_batches = 0
            running_var_val = torch.zeros((), device=device)
            running_cov_val = torch.zeros((), device=device)
            per_component_metrics: dict[str, float] = {}
            encoder.eval()
            decoder.eval()
            preprocessor.eval()

            # The same noise every validation pass, so the epochs' validation
            # losses (and the best-epoch checkpoint) compare like with like.
            validation_generator = torch.Generator(device=device).manual_seed(
                training_config.seed
            )
            with torch.no_grad():
                for component_name, validation_loader in validation_loaders.items():
                    component_loss = torch.zeros((), device=device)
                    component_batches = 0
                    for val_samples in validation_loader:
                        if (
                            args.max_val_batches is not None
                            and component_batches >= args.max_val_batches
                        ):
                            break
                        val_samples.to_(device)
                        validation_step = denoising_step(
                            val_samples,
                            preprocessor=preprocessor,
                            encoder=encoder,
                            decoder=decoder,
                            noise_level=noise_level,
                            generator=validation_generator,
                        )

                        if vicreg_cfg.enabled:
                            var_loss_val, cov_loss_val = vicreg_descriptor_loss(
                                validation_step.molecular_descriptor.flat,
                                target_std=vicreg_cfg.target_std,
                            )
                            running_var_val += var_loss_val
                            running_cov_val += cov_loss_val

                        component_loss += validation_step.denoising_loss.detach()
                        component_batches += 1

                    validation_loss_sum += component_loss
                    validation_batches += component_batches
                    per_component_metrics[f"denoising_val_loss/{component_name}"] = (
                        component_loss / (max(component_batches, 1) * normaliser)
                    ).item()

                # Every validation batch weighs the same, whichever component
                # it comes from.
                avg_validation_loss = (
                    validation_loss_sum / (max(validation_batches, 1) * normaliser)
                ).item()

                extra_metrics: dict[str, float] = {
                    "descriptor_norm_max_train": float(running_zmax.item()),
                    **per_component_metrics,
                }
                if vicreg_cfg.enabled:
                    extra_metrics["vicreg_variance_train"] = float(
                        (running_var / (batch_index + 1)).item()
                    )
                    extra_metrics["vicreg_covariance_train"] = float(
                        (running_cov / (batch_index + 1)).item()
                    )
                    extra_metrics["vicreg_variance_val"] = float(
                        (running_var_val / max(validation_batches, 1)).item()
                    )
                    extra_metrics["vicreg_covariance_val"] = float(
                        (running_cov_val / max(validation_batches, 1)).item()
                    )

                telemetry.log_pretraining_epoch(
                    epoch,
                    avg_train_loss,
                    avg_validation_loss,
                    current_lr=lr_scheduler.get_last_lr()[0],
                    extra_metrics=extra_metrics,
                )

                if telemetry.best_epoch:
                    # The bundle holds the uncompiled encoder, which shares its
                    # parameters with the torch.compile wrapper trained above.
                    save_checkpoint(bundle, training_data_dir)

            # Throw out the partial window so the first window of the next
            # training epoch isn't polluted with validation/checkpointing time.
            window_atoms = 0
            window_t0 = perf_counter()

    pyaml.to_yaml_file(
        training_data_dir / ARCHITECTURE_FILENAME,
        architecture_config,
    )


if __name__ == "__main__":
    main()
