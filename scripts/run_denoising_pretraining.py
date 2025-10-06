import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pydantic_yaml as pyaml
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import (
    TrainingConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.data_handling.sample import (
    PreprocessedSample,
    pretraining_padded_collate_fn,
)
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.data import (
    DatasetSplitting,
    worker_init_fn,
)
from threedscriptors.training.data.data_normalization import DataNormalizationModule
from threedscriptors.training.noise_scheduler import ConstantSchedule, NoiseModule
from threedscriptors.training.pretraining import atom_denoising_loss
from threedscriptors.training.telemetry import TrainingTelemetry

device = "cuda" if torch.cuda.is_available() else "cpu"




def parse_args():
    """
    Parse command-line arguments and return the run_name.
    """
    parser = argparse.ArgumentParser(
        description="Parse the --run_name argument for naming runs"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing the training config and architecture config file.",
    )

    args = parser.parse_args()
    return args.train_dir


def main():
    training_dir = parse_args()
    training_dir = Path(training_dir)

    torch.manual_seed(0)
    np.random.seed(0)

    training_config = pyaml.parse_yaml_file_as(
        TrainingConfig, f"{training_dir}/training_config.yaml"
    )
    architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig,
        f"{training_dir}/architecture_config.yaml",
    )

    full_dataset = MoleculeDataset.open_existing_dataset_from_dir(
        training_config.dataset_path
    )
    ds = TrainingMoleculeDataset(training_config.dataset_path, get_item=pos_emb_getitem)

    splitting = DatasetSplitting(full_dataset)
    train_idx, val_idx, split_name = next(
        splitting.get_split(training_config.split_config)
    )
    train_dataset = Subset(ds, train_idx)
    valid_dataset = Subset(ds, val_idx)

    training_idx = len(list(training_dir.glob("*/")))
    now = datetime.now()
    training_identifier = (
        f"{training_idx}-{now.strftime('%Y_%m_%d_%H_%M_%S')}-{split_name}"
    )
    training_data_dir = training_dir / Path(training_identifier)
    os.makedirs(training_data_dir)

    noise_scheduler = ConstantSchedule(training_config.noise_level)
    noise_module = NoiseModule(noise_scheduler)

    training_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8,
        persistent_workers=True,
        pin_memory=True,
        num_workers=32,
        shuffle=True,
        collate_fn=pretraining_padded_collate_fn,
    )

    validation_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8,
        persistent_workers=True,
        pin_memory=True,
        num_workers=16,
        shuffle=True,
        collate_fn=pretraining_padded_collate_fn,
    )

    dn = DataNormalizationModule(train_dataset)

    inv_mean_per_dim, inv_std_per_dim = dn.get_atomic_embedding_normalization_constants(
        irreps=full_dataset.config.irreps
    )

    mb = ModelBuilder(architecture_config=architecture_config)
    preprocessor = mb.build_preprocessor(inv_mean_per_dim, inv_std_per_dim)
    encoder = mb.build_encoder()
    decoder = mb.build_decoder()

    preprocessor.to(dtype=torch.float64)

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

    best_model_path = f"{training_data_dir}/best_model.pth"

    architecture_config.encoder_config.reload_state_dict = (
        f"{training_data_dir}/encoder.pth"
    )
    architecture_config.embedding_preprocess_config.reload_state_dict = (
        f"{training_data_dir}/atomic_preprocessor.pth"
    )
    architecture_config.positional_encoding_config.reload_state_dict = (
        f"{training_data_dir}/geometric_preprocessor.pth"
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
    ) as telemetry:
        encoder.to(device)
        decoder.to(device)
        preprocessor.to(device)

        print("Training Start")
        for epoch in range(training_config.epochs):
            # Initialize task and total train losses

            denoising_loss_accumulated = 0.0

            encoder.train()
            decoder.train()
            preprocessor.train()

            optimizer.zero_grad()

            for batch_index, samples in enumerate(training_loader):
                samples.to_(device)
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
                molecular_descriptor.register_hook(telemetry.get_track_grad_norm_fn())

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

                denoising_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    all_params, max_norm=training_config.max_grad_norm
                )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                denoising_loss_accumulated += denoising_loss.item()

            avg_train_loss = denoising_loss_accumulated / (
                (batch_index + 1)
                * architecture_config.embedding_preprocess_config.output_irreps_dim
            )

            accumulated_validation_loss = 0.0
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

                    accumulated_validation_loss += denoising_loss.item()

                avg_validation_loss = accumulated_validation_loss / (
                    (batch_idx + 1)
                    * architecture_config.embedding_preprocess_config.output_irreps_dim
                )

                telemetry.log_pretraining_epoch(
                    epoch, avg_train_loss, avg_validation_loss, current_lr=lr_scheduler.get_last_lr()
                )

                if telemetry.best_epoch:
                    torch.save(encoder.state_dict(), f"{training_data_dir}/encoder.pth")
                    torch.save(
                        preprocessor.atomic_preprocessor.state_dict(),
                        f"{training_data_dir}/atomic_preprocessor.pth",
                    )
                    torch.save(
                        preprocessor.geometric_preprocessor.state_dict(),
                        f"{training_data_dir}/geometric_preprocessor.pth",
                    )

    pyaml.to_yaml_file(
        training_data_dir / "post_training_architecture_config.yaml",
        architecture_config,
    )


if __name__ == "__main__":
    main()
