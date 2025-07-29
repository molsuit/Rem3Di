import argparse
import os
from datetime import datetime
from pathlib import Path
import math
import numpy as np
import pydantic_yaml as pyaml
import torch
from threedscriptors.model.remedi_model import REM3DIModel
from threedscriptors.data_handling.sample import PreprocessedSample
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from threedscriptors.data_handling.indexed_subset import IndexedSubset

from threedscriptors.training.dataset_splitting import (
    DatasetSplitting,
    SplitConfig,
    SplitStrategy,
)

from threedscriptors.evaluation.evaluation_pipeline import (
    DescriptorPCATask,
    EvalPipelineRunner,
)
from threedscriptors.evaluation.clustering import UMAPCalculator
from threedscriptors.configuration.data_config import DatasetSplit


from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.pretraining import atom_denoising_loss

from threedscriptors.training.telemetry import TrainingTelemetry
from threedscriptors.training.data_normalization import DataNormalizationModule
from threedscriptors.training.noise_scheduler import ConstantSchedule, NoiseModule


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """
    Parse command-line arguments and return the run_name.
    """
    parser = argparse.ArgumentParser(
        description="Parse the --run_name argument for naming runs"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run (e.g., experiment identifier)",
    )
    args = parser.parse_args()
    return args.run_name


run_name = parse_args()
torch.manual_seed(0)
np.random.seed(0)

training_run_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs"
)
training_idx = len(list(training_run_dir.glob("*/")))
now = datetime.now()
training_data_dir = training_run_dir / Path(
    f"{training_idx}-{now.strftime("%Y_%m_%d_%H_%M_%S")}-{run_name}"
)

os.makedirs(training_data_dir)

split_config = SplitConfig(
    strategy=SplitStrategy.SINGLE, N_folds=None, N_repeats=None, shuffle=True
)

training_config = TrainingConfig(
    batch_size=64,
    epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-3,
    max_grad_norm=1.0,
    wandb_active=True,
    split_config=split_config,
    training_data_dir=training_data_dir,
    mace_model_path="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/geomfull",
    noise_level=0.3,
    model_dir="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/geomfull",
    normalized_targets=True,
)

architecture_config = pyaml.parse_yaml_file_as(
    ArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)

dataset = reload_dataset_pipeline(training_config.dataset_path).build()

#dataset = dataset.convert_to_dataset_type(AtomicEmbeddingWithPositionsDataset)

dataset_splitting = DatasetSplitting(dataset)

lowest_val_losses = []

train_idx, val_idx, split_name = next(
    dataset_splitting.get_split(training_config.split_config)
)
print("Split")

os.makedirs(f"{training_config.training_data_dir}/{split_name.lower()}")

train_dataset = IndexedSubset(dataset, train_idx)
valid_dataset = IndexedSubset(dataset, val_idx)



noise_scheduler = ConstantSchedule(training_config.noise_level)
noise_module = NoiseModule(noise_scheduler)

training_loader = DataLoader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=sample_collate_fn,
)
validation_loader = DataLoader(
    valid_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    collate_fn=sample_collate_fn,
)


data_normalization = DataNormalizationModule(dataset=train_dataset)
print("Starting Normalization ")
inv_mean_per_dim, inv_std_per_dim = (
    data_normalization.get_atomic_embedding_normalization_constants()
)

print("Normalized")

print("buuilding model")
mb = ModelBuilder(architecture_config=architecture_config)
preprocessor = mb.build_preprocessor(inv_mean_per_dim, inv_std_per_dim)
encoder = mb.build_encoder()
decoder = mb.build_decoder()

preprocessor.to(dtype=torch.float64)

config = {
    "architecture_config": architecture_config.model_dump(),
    "training_config": training_config.model_dump(),
    "dataset_config": dataset.dataset_config.model_dump(),
}
print("Model built")

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

best_model_path = f"{training_config.training_data_dir}/best_model.pth"


with TrainingTelemetry(
    training_config=training_config,
    dataset_config=dataset.dataset_config,
    run_name=run_name,
    split_name=split_name,
    config=config,
) as telemetry:

    encoder.to(device)
    decoder.to(device)
    preprocessor.to(device)

    print("Starting Training")

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

            denoised_embeddings = decoder(
                noised_embeddings, molecular_descriptor, samples.padding_mask
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
                preprocessed_val_samples: PreprocessedSample = preprocessor(val_samples)

                input_atomic_embeddings = (
                    preprocessed_val_samples.preprocessed_atomic_embeddings.clone()
                )

                noised_embeddings = noise_module(
                    preprocessed_val_samples.preprocessed_atomic_embeddings,
                    preprocessed_val_samples.padding_mask,
                )

                molecular_descriptor = encoder(preprocessed_val_samples)

                denoised_embeddings = decoder(
                    noised_embeddings, molecular_descriptor, val_samples.padding_mask
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

            telemetry.log_pretraining_epoch(epoch, avg_train_loss, avg_validation_loss)


# package everything into a remedi model

torch.save(encoder.state_dict(), f"{training_config.training_data_dir}/encoder.pth")
torch.save(preprocessor.atomic_preprocessor.state_dict(), f"{training_config.training_data_dir}/atomic_preprocessor.pth")
torch.save(preprocessor.geometric_preprocessor.state_dict(), f"{training_config.training_data_dir}/geometric_preprocessor.pth")

model = REM3DIModel(preprocessor=preprocessor, encoder=encoder)



umap_clustering_calculator = UMAPCalculator()
umap_task_train = DescriptorPCATask(train_dataset, umap_clustering_calculator)
umap_task_val = DescriptorPCATask(valid_dataset, umap_clustering_calculator)
eval_train = EvalPipelineRunner([umap_task_train], "pcqmc", DatasetSplit.TRAIN)
eval_validation = EvalPipelineRunner([umap_task_val], "pcqm", DatasetSplit.VALIDATION)


eval_train.evaluate(model)
eval_validation.evaluate(model)
train_figs, train_result_report = eval_train.output_results(
    output_directory=f"{training_config.training_data_dir}/trainset_results",
    model_name=run_name,
)

validation_figs, validation_result_report = eval_validation.output_results(
    output_directory=f"{training_config.training_data_dir}/valset_results",
    model_name=run_name,
)
