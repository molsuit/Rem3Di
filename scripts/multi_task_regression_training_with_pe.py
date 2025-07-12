import argparse
import os
from datetime import datetime
from pathlib import Path
import math
import numpy as np
import pydantic_yaml as pyaml
import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from threedscriptors.data_handling.dataset import RegressionDatasetwithPositions
from threedscriptors.training.dataset_splitting import DatasetSplitting, SplitConfig, SplitStrategy

from threedscriptors.training.training_utils import cosine_matrix
from threedscriptors.data_handling.dataset_analysis import DatasetPostLoadAnalysis
import wandb
from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.evaluation.training_evaluation import regression_pipeline, chiral_regression_pipeline
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.regression_training import multitask_masked_loss, DynamicallyWeighedMultitaskLoss, BaseMultitaskLoss

from threedscriptors.training.data_normalization import DataNormalizationModule

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
    strategy= SplitStrategy.REPEATED_CV,
    N_folds = 5,
    N_repeats = 5,
    shuffle = True
)

training_config = TrainingConfig(
    batch_size=64,
    epochs=400,
    learning_rate=1e-5,
    weight_decay=1e-3,
    max_grad_norm=1.0,
    wandb_active=True,
    split_config=split_config,
    training_data_dir=training_data_dir,
    mace_model_path="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet",
    test_dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_test",
    model_dir="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/antiviral_admet",
    normalized_targets=True,
)

architecture_config = pyaml.parse_yaml_file_as(
    ArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)

dataset = reload_dataset_pipeline(training_config.dataset_path).build()

dataset = dataset.convert_to_dataset_type(RegressionDatasetwithPositions)

dataset_splitting = DatasetSplitting(dataset)

for train_idx, val_idx in dataset_splitting.get_split(training_config.split_config):


    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)



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


    data_normalization = DataNormalizationModule(dataset = train_dataset)
    inv_mean_per_dim, inv_std_per_dim = data_normalization.get_atomic_embedding_normalization_constants()



    mb = ModelBuilder(architecture_config=architecture_config)
    model = mb.build_model(
        mean_atomic_embedding=inv_mean_per_dim,
        std_atomic_embedding=inv_std_per_dim
    )




    print(f"Trainable Parameters: {mb.N_trainable_parameters}")


    config = {
        "architecture_config": architecture_config.model_dump(),
        "training_config": training_config.model_dump(),
        "dataset_config": dataset.dataset_config.model_dump(),
    }


    if training_config.wandb_active:
        wandb.init(
            project="threedscriptors",
            entity="threedscriptors",
            group=
            name=run_name,
            config=config,
        )

        #wandb.watch(models=[model.preprocessor,model.multitask_heads], log="all", log_freq=30)

        
    model.to(device)



    num_opt_steps = training_config.epochs * len(training_loader)


    #loss_fn = DynamicallyWeighedMultitaskLoss(N_tasks = len(train_dataset.dataset_config.tasks))

    loss_fn = BaseMultitaskLoss()




    optimizer = optim.AdamW(
        [{"params" : model.parameters(), "lr" : training_config.learning_rate, "weight_decay" : training_config.weight_decay},
        #{"params": loss_fn.parameters(),  "lr": 1e-4, "weight_decay" : 0.0}
        ]
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=[training_config.learning_rate], total_steps=num_opt_steps
    )

    task_names = dataset.dataset_config.get_task_names()
    stds_per_task = data_normalization.std_tasks
    #assert model.multitask_heads.task_heads.keys() == stds_per_task.keys()

    print("Starting Training")

    loss_data = []
    best_val_loss = math.inf
    best_model_path = f"{training_config.training_data_dir}/best_model.pth"


    loss_fn.to(device)


    shared_parameters = [p for p in model.encoder.parameters() if p.requires_grad]



    rows, cols = torch.tril_indices(row=len(task_names), col=len(task_names), offset=-1)


    task_pair_indices = torch.nonzero(torch.tril(torch.ones((len(task_names),len( task_names))),-1))

    map_task_to_sim_dict = {idx : (task_names[p[0]],task_names[p[1]]) for idx, p in enumerate(task_pair_indices)}

    full_training_grad_alignment = []

    for epoch in range(training_config.epochs):
        running_tloss = 0.0
        weighed_loss_per_task_train = torch.zeros(
            len(dataset.dataset_config.tasks), device=device
        )

        loss_fn.train()
        model.train()
        optimizer.zero_grad()

        #grad_alignment_epoch = []

        for _batch, samples in enumerate(training_loader):
            samples = data_normalization(samples)
            samples.to_(device)



            model_output = model(samples)


            loss, batch_weighed_loss_per_task_train = loss_fn(samples, model_output)


            #grad_cosine_matrix = cosine_matrix(batch_weighed_loss_per_task_train, shared_parameters)
            
            #grad_alignment_epoch.append(grad_cosine_matrix[rows, cols].detach().cpu())

            weighed_loss_per_task_train += batch_weighed_loss_per_task_train.detach()


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=training_config.max_grad_norm
            )



            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_tloss += loss.item()

        avg_tloss = running_tloss / (
            _batch + 1
        )  
        weighed_loss_per_task_train = weighed_loss_per_task_train / (_batch + 1)

        running_vloss = 0.0
        running_mean_val_loss = 0.0

        weighed_loss_per_task_val = torch.zeros(
            len(dataset.dataset_config.tasks), device=device
        )
        

        loss_fn.eval()
        model.eval()
        with torch.no_grad():
            for _batch, val_samples in enumerate(validation_loader):
                

                val_samples = data_normalization(val_samples)
                val_samples.to_(device)

                val_output = model(val_samples)

                loss, batch_weighed_loss_per_task_val = loss_fn(val_samples, val_output)
                weighed_loss_per_task_val += batch_weighed_loss_per_task_val

                running_vloss += loss.item()

                #val_base_loss, _ = val_loss_fn(val_samples, val_output)
                #running_mean_val_loss += val_base_loss.item() 

            avg_vloss = running_vloss / (_batch + 1)
            weighed_loss_per_task_val = weighed_loss_per_task_val / (_batch + 1)

            print(f"Epoch {epoch} Training Loss: {avg_tloss} Validation Loss: {avg_vloss}")

            if running_vloss < best_val_loss:
                best_val_loss = running_vloss
                torch.save(model.state_dict(), best_model_path)
                print(f"  - New best model (val_loss {avg_vloss:.4f}), saving to {best_model_path}")



            current_lr = scheduler.get_last_lr()

            destandardized_loss_per_task_val = (
                weighed_loss_per_task_val.cpu().detach().numpy() #* stds
            )

            val_dict = dict(
                zip(
                    task_names,
                    destandardized_loss_per_task_val.tolist(),
                    strict=False,
                )
            )

            destandardized_loss_per_task_train = (
                weighed_loss_per_task_train.cpu().detach().numpy() #* stds
            )

            train_dict = dict(
                zip(
                    task_names,
                    destandardized_loss_per_task_train.tolist(),
                    strict=False,
                )
            )

    #        task_loss_weights = dict(zip(task_names, loss_fn.log_vars.cpu().detach().numpy().tolist()))
    #       

            #grad_alignment_epoch = torch.stack(grad_alignment_epoch, dim = -1)
            #print(f"Gradalginemnet shape {grad_alignment_epoch.shape}")
            #full_training_grad_alignment.append(grad_alignment_epoch)

    #
            task_pair_string_list = [str(v) for v in map_task_to_sim_dict.values() ]
            #alignment_dict = dict(zip(task_pair_string_list, grad_alignment_epoch.#mean(dim=0)))


            epoch_loss_dict = {
                "epoch": epoch,
                "validation_loss": avg_vloss,
                "train_loss": avg_tloss,
                "task_validation_loss": val_dict,
                "task_train_loss": train_dict,
                "learning_rate": current_lr[0],
                #"grad_alignment": alignment_dict
                #"loss_weights": task_loss_weights

            }

            loss_data.append(epoch_loss_dict)

            if training_config.wandb_active:
                wandb.log(epoch_loss_dict)



    #
    #
    #print(map_task_to_sim_dict)
    #
    #
    #cosine_sim_dict = {v : [] for v in map_task_to_sim_dict.values()}
    #print(cosine_sim_dict)
    #
    #
    #full_training_grad_alignment = torch.stack(full_training_grad_alignment)
    #print(full_training_grad_alignment.shape)
    #
    #
    #import matplotlib.pyplot as plt
    #
    #
    #
    #times = list(range(training_config.epochs))
    #
    #for i, task_pair in map_task_to_sim_dict.items():
    #
    #
    #    task_pair_data= full_training_grad_alignment[:,i,:].squeeze()
    #    
    #
    #    print(task_pair_data.shape)
    #    print(times)
    #
    #    fig, ax = plt.subplots()
    #
    #    fig.set_figwidth(20.)

    #    ax.violinplot(task_pair_data.T, positions=times, widths=0.8, #showmeans=False, showextrema=True, showmedians=True)
    #    ax.set_xlabel("Time")
    #    ax.set_ylabel("Cosine Similarity of gradients")
    #    ax.set_title(f"Distribution grad cosine sim over training time {task_pair}")

    #    fig.savefig(f"{training_config.training_data_dir}/grad_tasks_cosine_sim_{str#(task_pair)}.png")
    #
    #
    #    print(task_pair)
    #    ratio_0 = (task_pair_data < 0.0).float().mean()
    #    ratio_0_1 = (task_pair_data < -0.1).float().mean() 
    #    print(f"Smaller 0.0  {ratio_0}")
    #    print(f"Smaller -0.1  {ratio_0_1}")
    #
    #
    #
    #
    #
    ## undo the normalization ??
    final_loss = avg_vloss
    print(final_loss)
    #
    print("Loading best model from", best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()



    torch.save(model.state_dict(), f"{training_config.training_data_dir}/regression_model.pth")
    #
    torch.save(
        model.preprocessor.state_dict(), f"{training_config.training_data_dir}/preprocessor.pth"
    )
    torch.save(model.encoder.state_dict(), f"{training_config.training_data_dir}/encoder.pth")
    #
    pyaml.to_yaml_file(
        f"{training_config.training_data_dir}/training_config.yaml", training_config
    )


    architecture_config.reload_full_model_weights = (
        f"{training_config.training_data_dir}/regression_model.pth"
    )

    pyaml.to_yaml_file(
        f"{training_config.training_data_dir}/architecture_config.yaml", architecture_config
    )


    pyaml.to_yaml_file(
        f"{training_config.training_data_dir}/dataset_config.yaml",
        dataset.dataset_config,
    )



    with open(f"{training_config.training_data_dir}/training_losses.yaml", "x") as f:
        yaml.safe_dump(loss_data, f)

    figs = {}


    training_evaluation_pipeline = regression_pipeline(train_dataset)
    training_evaluation_pipeline.evaluate(model)
    train_figs, train_result_report = training_evaluation_pipeline.output_results(
        output_directory=f"{training_config.training_data_dir}/trainset_results", model_name=run_name
    )




    validation_evaluation_pipeline = regression_pipeline(valid_dataset)
    validation_evaluation_pipeline.evaluate(model)
    train_figs, train_result_report =validation_evaluation_pipeline.output_results(
        output_directory=f"{training_config.training_data_dir}/valset_results", model_name=run_name
    )

    test_pipeline_orchestrator = reload_dataset_pipeline(
        training_config.test_dataset_path,
        mean_targets=mean_target_per_task,
        std_targets=std_target_per_task,
    )
    test_dataset = test_pipeline_orchestrator.build()
    test_dataset.expand_embedding_num_atoms(new_max_num_atoms=train_dataset.dataset_config.max_atoms)


    test_evaluation_pipeline = regression_pipeline(test_dataset)
    test_evaluation_pipeline.evaluate(model)
    figs, result_report = test_evaluation_pipeline.output_results(
        output_directory=f"{training_config.training_data_dir}/testset_results", model_name=run_name
    )

    if training_config.wandb_active:
        for figname, figure in figs.items():
            wandb.log({figname: figure})
