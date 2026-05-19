import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pydantic_yaml as pyaml
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import (
    TrainingConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from threedscriptors.data_handling.sample import (
    PreprocessedSample,
    yield_molecules_collate_fn,
)
from threedscriptors.training.data import (
    DatasetSplitting,
)
from threedscriptors.training.data.samplers import lengths_from_ptr
from threedscriptors.training.noise_scheduler import ConstantSchedule, NoiseModule
from threedscriptors.training.pretraining import (
    atom_denoising_loss,
    vicreg_descriptor_loss,
)
from threedscriptors.training.telemetry import TrainingTelemetry
import logging

from torch.profiler import profile, ProfilerActivity

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch._dynamo
torch._logging.set_logs(recompiles=True)

# nvalchemiops applies @torch.compile to prepare_batch_idx_ptr, which is on the
# MACE neighbor-list path called every step. Bucket sampling makes the input
# shapes (total atoms, num molecules) vary, triggering repeated recompiles. The
# function does ~µs of cumsum/bincount work — not worth compiling. Override the
# decorated symbol with a dynamo-disabled version before any model is built.
import nvalchemiops.torch.neighbors.neighbor_utils as _nv_neighbor_utils
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
        "--dataset_path",
        type=Path,
        default=None,
        help="Optional override for the dataset path (e.g. compute-node-staged data).",
    )
    return parser.parse_args()

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



def main():
    args = parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    training_config = pyaml.parse_yaml_file_as(TrainingConfig, args.training_config)
    architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig, training_config.model_config_path
    )

    dataset_path = args.dataset_path if args.dataset_path is not None else training_config.dataset_path

    full_dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_path)
    ds = TrainingMoleculeDataset(dataset_path, get_item=atoms_getitem, in_memory=True)

    splitting = DatasetSplitting(full_dataset)
    train_idx, val_idx, split_name = next(
        splitting.get_split(training_config.split_config)
    )
    train_dataset = Subset(ds, train_idx)
    valid_dataset = Subset(ds, val_idx)

    all_lengths = lengths_from_ptr(np.asarray(full_dataset.ptr[:]))
    train_lengths = all_lengths[np.asarray(train_idx, dtype=np.int64)]
    val_lengths = all_lengths[np.asarray(val_idx, dtype=np.int64)]

    if training_config.training_directory is not None:
        training_data_dir = training_config.training_directory
        training_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_base = training_config.output_base
        output_base.mkdir(parents=True, exist_ok=True)
        training_idx = len(list(output_base.glob("*/")))
        now = datetime.now()
        training_identifier = (
            f"{training_idx}-{now.strftime('%Y_%m_%d_%H_%M_%S')}-{split_name}"
        )
        training_data_dir = output_base / training_identifier
        training_data_dir.mkdir()

    logger = setup_logging(filename=training_data_dir / "logfile.info")
    logger.info("Loaded Dataset")

    noise_scheduler = ConstantSchedule(training_config.noise_level)
    noise_module = NoiseModule(noise_scheduler)

    training_loader = training_config.dataloader.build(
        train_dataset,
        lengths=train_lengths,
        collate_fn=yield_molecules_collate_fn,
        shuffle=True,
    )
    validation_loader = training_config.dataloader.build(
        valid_dataset,
        lengths=val_lengths,
        collate_fn=yield_molecules_collate_fn,
        shuffle=False,
    )

    logger.info("Data Loaders Prepared")

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
        training_data_dir / "post_training_architecture_config.yaml",
        architecture_config,
    )


    with TrainingTelemetry(
        wandb_active=training_config.wandb_active,
        run_name=training_config.training_name,
        group_name=training_config.run_group,
        out_dir=training_data_dir,
        config = {"train_config": training_config.model_dump(), "architecture_config": architecture_config.model_dump()}
    ) as telemetry:
        
        encoder.to(device, dtype = torch.float32)
        decoder.to(device, dtype = torch.float32)
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
                samples.to_(device)
                # Capture before preprocessor — it mutates atomic_positions from
                # flat (total_atoms, 3) to padded (B, N_max, 3).
                batch_atoms = int(samples.atomic_positions.shape[0])
                preprocessed_samples: PreprocessedSample = preprocessor(samples)

                input_atomic_embeddings = (
                    preprocessed_samples.preprocessed_atomic_embeddings.clone()
                )

                noised_embeddings = noise_module(
                    preprocessed_samples.preprocessed_atomic_embeddings,
                    preprocessed_samples.padding_mask,
                )

                noise_module.step()

                molecular_descriptor = encoder(preprocessed_samples)

                noised_preprocessing_sample = PreprocessedSample(
                    preprocessed_atomic_embeddings=noised_embeddings,
                    initial_pair_representation=preprocessed_samples.initial_pair_representation.detach(),
                    geometrical_encoding=preprocessed_samples.geometrical_encoding.detach(),
                    padding_mask=preprocessed_samples.padding_mask,
                    pair_mask=preprocessed_samples.pair_mask,
                )

                denoised_embeddings = decoder(
                    noised_preprocessing_sample, molecular_descriptor
                )

                noise_level = noise_scheduler.value
                denoising_loss = atom_denoising_loss(
                    input_atomic_embeddings,
                    denoised_embeddings,
                    padding_mask=samples.padding_mask,
                    noise_level=noise_level,
                )

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

            avg_train_loss = (running / (
                (batch_index + 1)
                * architecture_config.embedding_preprocess_config.output_irreps_dim
            )).item()

            running = torch.zeros((), device=device)
            running_var_val = torch.zeros((), device=device)
            running_cov_val = torch.zeros((), device=device)
            encoder.eval()
            decoder.eval()
            preprocessor.eval()

            with torch.no_grad():
                for batch_idx, val_samples in enumerate(validation_loader):
                    val_samples.to_(device)
                    preprocessed_val_samples: PreprocessedSample = preprocessor(
                        val_samples
                    )

                    input_atomic_embeddings = (
                        preprocessed_val_samples.preprocessed_atomic_embeddings.clone()
                    )

                    noised_embeddings = noise_module(
                        preprocessed_val_samples.preprocessed_atomic_embeddings,
                        preprocessed_val_samples.padding_mask,
                    )

                    molecular_descriptor = encoder(preprocessed_val_samples)

                    if vicreg_cfg.enabled:
                        var_loss_val, cov_loss_val = vicreg_descriptor_loss(
                            molecular_descriptor.flat,
                            target_std=vicreg_cfg.target_std,
                        )
                        running_var_val += var_loss_val
                        running_cov_val += cov_loss_val

                    noised_preprocessing_sample = PreprocessedSample(
                        preprocessed_atomic_embeddings=noised_embeddings,
                        initial_pair_representation=preprocessed_val_samples.initial_pair_representation.detach(),
                        geometrical_encoding=preprocessed_val_samples.geometrical_encoding.detach(),
                        padding_mask=preprocessed_val_samples.padding_mask,
                        pair_mask=preprocessed_val_samples.pair_mask,
                    )

                    denoised_embeddings = decoder(
                        noised_preprocessing_sample, molecular_descriptor
                    )

                    noise_level = noise_scheduler.value
                    denoising_loss = atom_denoising_loss(
                        input_atomic_embeddings,
                        denoised_embeddings,
                        padding_mask=val_samples.padding_mask,
                        noise_level=noise_level,
                    )

                    running += denoising_loss.detach()

                avg_validation_loss = (running / (
                    (batch_idx + 1)
                    * architecture_config.embedding_preprocess_config.output_irreps_dim
                )).item()

                extra_metrics: dict[str, float] = {
                    "descriptor_norm_max_train": float(running_zmax.item()),
                }
                if vicreg_cfg.enabled:
                    extra_metrics["vicreg_variance_train"] = float(
                        (running_var / (batch_index + 1)).item()
                    )
                    extra_metrics["vicreg_covariance_train"] = float(
                        (running_cov / (batch_index + 1)).item()
                    )
                    extra_metrics["vicreg_variance_val"] = float(
                        (running_var_val / (batch_idx + 1)).item()
                    )
                    extra_metrics["vicreg_covariance_val"] = float(
                        (running_cov_val / (batch_idx + 1)).item()
                    )

                telemetry.log_pretraining_epoch(
                    epoch,
                    avg_train_loss,
                    avg_validation_loss,
                    current_lr=lr_scheduler.get_last_lr()[0],
                    extra_metrics=extra_metrics,
                )

                if telemetry.best_epoch:
                    encoder_to_save = getattr(encoder, "_orig_mod", encoder)
                    torch.save(encoder_to_save.state_dict(), f"{training_data_dir}/encoder.pth")
                    torch.save(
                        preprocessor.atomic_preprocessor.state_dict(),
                        f"{training_data_dir}/atomic_preprocessor.pth",
                    )
                    torch.save(
                        preprocessor.geometric_preprocessor.state_dict(),
                        f"{training_data_dir}/geometric_preprocessor.pth",
                    )

            # Throw out the partial window so the first window of the next
            # training epoch isn't polluted with validation/checkpointing time.
            window_atoms = 0
            window_t0 = perf_counter()

    pyaml.to_yaml_file(
        training_data_dir / "post_training_architecture_config.yaml",
        architecture_config,
    )


if __name__ == "__main__":
    main()
