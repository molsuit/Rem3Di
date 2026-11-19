import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, Subset

from remedi.configuration.architecture_config import (
    ArchitectureConfig,
)
from remedi.configuration.training_config import PretrainingConfig, ProbeConfig
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, save_checkpoint
from remedi.model.preprocessing.invariant_statistics import (
    ensure_invariant_statistics,
)
from remedi.training.data.mixture import (
    MixtureDatasets,
    resolve_mixture,
    stage_datasets,
)
from remedi.training.data.sample import yield_molecules_collate_fn
from remedi.training.data.samplers import BucketBatchSampler
from remedi.training.data.training_dataset import atoms_getitem
from remedi.training.linear_probe import LinearProbeMonitor
from remedi.training.pretraining import denoising_step, descriptor_statistics
from remedi.training.telemetry import ThroughputWindow, TrainingTelemetry
from remedi.training.training_state import (
    TRAINING_STATE_FILENAME,
    load_training_state,
    save_training_state,
)
from remedi_data.yaml_io import read_model, write_model

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch._dynamo  # noqa: E402 - after the device and logging setup

torch._logging.set_logs(recompiles=True)


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
    parser.add_argument(
        "--stage_directory",
        type=Path,
        default=None,
        help=(
            "Copy the datasets' files into this directory (node-local disk) "
            "before training and read them from there."
        ),
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
    """Log to ``filename``, appending: a resumed run continues the same log."""
    logger = logging.getLogger("remedi")
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.FileHandler(filename=filename, mode="a")
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
        logger.propagate = False
    return logger


def main():  # noqa: C901 - one training loop
    args = parse_args()

    training_config = read_model(args.training_config, PretrainingConfig)
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    if args.max_epochs is not None:
        training_config.epochs = args.max_epochs
    architecture_config = read_model(
        training_config.model_config_path, ArchitectureConfig
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
    if args.stage_directory is not None:
        staged = stage_datasets(data, args.stage_directory)
        data = staged.data
        logger.info(
            "Staged %.2f GiB of %d datasets to %s in %.1f s",
            staged.copied_bytes / 2**30,
            len(staged.staged_directories),
            args.stage_directory,
            staged.seconds,
        )
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

    noise = training_config.noise
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

    # MACE is frozen: only the parameters that train go to the optimizer.
    all_params = [
        parameter
        for module in (encoder, decoder, preprocessor)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        all_params,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    total_steps = training_config.epochs * len(training_loader)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[training_config.learning_rate],
        total_steps=total_steps,
    )

    write_model(
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

        # Every trained module, as the training state saves and restores them.
        # (The preprocessor's MACE model is frozen and not saved.)
        state_modules = {
            "encoder": encoder,
            "decoder": decoder,
            "atomic_preprocessor": preprocessor.atomic_preprocessor,
            "geometric_preprocessor": preprocessor.geometric_preprocessor,
        }
        resumed = load_training_state(
            training_data_dir,
            modules=state_modules,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            telemetry=telemetry,
            map_location=device,
        )
        start_epoch, global_step = resumed if resumed is not None else (0, 0)
        if resumed is not None:
            if lr_scheduler.total_steps != total_steps:
                raise ValueError(
                    f"{training_data_dir} holds a run planned for "
                    f"{lr_scheduler.total_steps} steps, this config gives "
                    f"{total_steps}: a run resumes only with the epochs, data and "
                    "batching it was started with."
                )
            logger.info(
                "Resumed from %s: %d epochs and %d steps done",
                TRAINING_STATE_FILENAME,
                start_epoch,
                global_step,
            )

        # The denoising target is the standardised invariants: their statistics
        # are estimated once, before the first epoch, and saved with the weights
        # (a resumed run has loaded them above).
        statistics = ensure_invariant_statistics(
            preprocessor,
            training_loader,
            max_batches=training_config.normalization_batches,
            device=device,
        )
        if statistics is not None:
            logger.info(
                "Invariant statistics from %d atoms; %d constant channels",
                statistics.atom_count,
                statistics.constant_channel_count,
            )

        if compile_cfg.enabled:
            encoder = torch.compile(encoder, dynamic=True)
            decoder = torch.compile(decoder, dynamic=True)

        print("Training Start")

        # Throughput every WINDOW_STEPS batches: atoms/s, step time, the share
        # spent waiting for the loader, GPU utilisation and peak memory. The
        # first window includes compile warm-up and the workers' start.
        WINDOW_STEPS = 50
        window = ThroughputWindow(device)

        for epoch in range(start_epoch, training_config.epochs):
            running = torch.zeros((), device=device)
            training_batches = 0

            encoder.train()
            decoder.train()
            preprocessor.train()

            optimizer.zero_grad()

            # Epoch e always draws the same batches, also in a resumed run and
            # whatever iterated the loader before (the statistics estimate).
            if isinstance(training_loader.batch_sampler, BucketBatchSampler):
                training_loader.batch_sampler.set_epoch(epoch)

            for batch_index, samples in enumerate(training_loader):
                if (
                    args.max_train_batches is not None
                    and batch_index >= args.max_train_batches
                ):
                    break
                window.batch_ready()
                samples.to_(device)
                step = denoising_step(
                    samples,
                    preprocessor=preprocessor,
                    encoder=encoder,
                    decoder=decoder,
                    noise=noise,
                )
                batch_atoms = step.atom_count
                denoising_loss = step.denoising_loss
                if not torch.isfinite(denoising_loss):
                    raise FloatingPointError(
                        f"Non-finite denoising loss ({denoising_loss.item()}) at "
                        f"epoch {epoch}, batch {batch_index}, step {global_step}. "
                        f"The last good state is {TRAINING_STATE_FILENAME}."
                    )

                denoising_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    all_params,
                    max_norm=training_config.max_grad_norm,
                    error_if_nonfinite=True,
                )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                running += denoising_loss.detach()
                training_batches += 1

                window.step_done(batch_atoms)
                global_step += 1
                if global_step % WINDOW_STEPS == 0:
                    throughput = {
                        "global_step": global_step,
                        "epoch_idx": epoch,
                        "denoising_train_loss_step": denoising_loss.item(),
                        **window.metrics(),
                    }
                    telemetry.log_throughput(throughput)
                    logger.info(
                        "step %d: %s",
                        global_step,
                        {k: round(v, 3) for k, v in throughput.items()},
                    )

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

            if training_batches == 0:
                raise RuntimeError("The training loader yielded no batches.")
            # Already a mean over atoms and channels: 1.0 is a decoder that
            # outputs zero, lower is learned structure.
            avg_train_loss = (running / training_batches).item()

            validation_loss_sum = torch.zeros((), device=device)
            # The same batches with every molecule given its neighbour's
            # descriptor: the excess over the real loss is what the decoder
            # takes from the descriptor.
            shuffled_loss_sum = torch.zeros((), device=device)
            shuffled_batches = 0
            matching_real_loss_sum = torch.zeros((), device=device)
            validation_batches = 0
            validation_descriptors: list[torch.Tensor] = []
            per_component_metrics: dict[str, float] = {}
            encoder.eval()
            decoder.eval()
            preprocessor.eval()

            # The same noise levels and noise every validation pass, so the
            # epochs' validation losses (and the best-epoch checkpoint) compare
            # like with like.
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
                            noise=noise,
                            generator=validation_generator,
                            with_shuffled_descriptor=True,
                        )
                        if validation_step.shuffled_descriptor_loss is not None:
                            shuffled_loss_sum += (
                                validation_step.shuffled_descriptor_loss
                            )
                            matching_real_loss_sum += validation_step.denoising_loss
                            shuffled_batches += 1
                        validation_descriptors.append(
                            validation_step.molecular_descriptor.flat.cpu()
                        )
                        component_loss += validation_step.denoising_loss.detach()
                        component_batches += 1

                    validation_loss_sum += component_loss
                    validation_batches += component_batches
                    per_component_metrics[f"denoising_val_loss/{component_name}"] = (
                        component_loss / max(component_batches, 1)
                    ).item()

                # Every validation batch weighs the same, whichever component
                # it comes from.
                avg_validation_loss = (
                    validation_loss_sum / max(validation_batches, 1)
                ).item()
                if not np.isfinite(avg_validation_loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss in epoch {epoch}."
                    )

                # Collapse check on the descriptors themselves, over the whole
                # validation set (training batches are too uniform in size).
                extra_metrics: dict[str, float] = dict(per_component_metrics)
                if shuffled_batches:
                    shuffled_loss = (shuffled_loss_sum / shuffled_batches).item()
                    extra_metrics["denoising_val_loss_shuffled_descriptor"] = (
                        shuffled_loss
                    )
                    extra_metrics["descriptor_gap"] = (
                        shuffled_loss
                        - (matching_real_loss_sum / shuffled_batches).item()
                    )
                all_descriptors = torch.cat(validation_descriptors)
                if all_descriptors.shape[0] >= 2:
                    extra_metrics.update(descriptor_statistics(all_descriptors))

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

            # After the checkpoint: a job killed from here on resumes at the
            # next epoch with the best-epoch files already in place.
            save_training_state(
                training_data_dir,
                completed_epochs=epoch + 1,
                global_step=global_step,
                modules=state_modules,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                telemetry=telemetry,
            )

            # A fresh window, so the next epoch's first window does not count
            # validation and checkpointing time.
            window = ThroughputWindow(device)


if __name__ == "__main__":
    main()
