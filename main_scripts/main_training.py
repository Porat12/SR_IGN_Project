import torch
import wandb
import shutil

import constants

from models.model_builder import Autoencoder
from utils.yaml_processing import register_yaml_constructors, read_yaml_config
from optimization_builders import build_optimizer, build_scheduler
from losses.loss_builder import build_loss

from loops.epochs_loop import epochs_loop
from loops.evaluation import test_loop

from dataset_classes.dataloader_builders import build_dataloaders

def main_training(config_path):

    print("\n")
    print("-"*80)
    register_yaml_constructors()
    print("-"*80)

    config = read_yaml_config(config_path)
    run_name = f"{ config["model"]["name"] }_{config["data"]["name"]}"
    print("-"*80)
    print("\n")

    print("-"*80)
    print("Starting SR-IGN training script...")
    print(f"Using {constants.device} device.")
    print("-"*80)
    print("\n")



    print("-"*80)
    wandb.login(key=constants.wandb_key) # For HPC, also works in local PC.

    wandb.init(
    project = "Super Resolution Using an Idempotent Neural Network",
    name = run_name,
    entity = "porat-hai-technion-israel-institute-of-technology",
    config = config
    )

    print("-"*80)
    print("\n")



    print("-"*80)

    training_config = config["training"]
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]


    model_config = config["model"]
    model = Autoencoder(model_config["encoder"], model_config["decoder"]).to(constants.device)
    model_copy = Autoencoder(model_config["encoder"], model_config["decoder"]).to(constants.device)

    print("Model architecture:\n")
    print(model)
    print("-"*80)
    print("\n")


    print("-"*80)
    print("Builders:")

    data_config = config["data"]
    train_loader, test_loader, valid_loader = build_dataloaders(batch_size, **data_config)
    eval_loader = test_loader if valid_loader is None else valid_loader
    if valid_loader is None:
        print("---- ---- --- No validation loader provided, using test loader for validation. not good ):\n")
    print("----  dataloaders built successfully.\n")

    optimizer = build_optimizer(config, model)
    print("----  optimizer built successfully.\n")


    scheduler = build_scheduler(optimizer, config)
    is_batch_scheduler = config["scheduler"]["is_batch_scheduler"]
    print(f"----  scheduler built successfully. is_batch_scheduler: {is_batch_scheduler}\n")


    train_loss, test_loss = build_loss(config)
    print(f"----  loss functions built successfully. Train loss: {train_loss.__name__}, Test loss: {test_loss.__name__}")
    print("-"*80)
    print("\n\n")
    loss_config = config["loss"]


    epochs_loop(epochs, train_loader, eval_loader, model, model_copy, train_loss, test_loss, optimizer, scheduler, is_batch_scheduler, **loss_config["params"])
    print("Training completed.")
    print("-"*80)
    print("\n")

    print("-"*80)
    print("logging artifact...")
    torch.save(model.state_dict(), "tmp_artifact_files/weights.pt")
    shutil.copy(config_path, "tmp_artifact_files/config.yaml")

    artifact_name = f"{wandb.run.name}.config_weights"
    artifact = wandb.Artifact(
                name = artifact_name,
                type = "config_weights",
                description = "Config, and final weights"
                )
    
    artifact.add_file("tmp_artifact_files/weights.pt")
    artifact.add_file("tmp_artifact_files/config.yaml")

    wandb.run.log_artifact(artifact)

    print("Artifact logged successfully.")
    print("-"*80)
    print("\n")
    
    print("-"*80)
    print("Test evaluation:")
    test_loop(test_loader, model, test_loss, is_test=True, **loss_config["params"])
    print("Test evaluation completed.")
    print("-"*80)
    print("\n")

    print("-"*80)
    wandb.finish()
    print("-"*80)
    print("\n")
    return artifact_name
