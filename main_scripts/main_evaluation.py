import torch
import wandb

import re

import constants

import matplotlib.pyplot as plt

from models.model_builder import Autoencoder
from utils.yaml_processing import register_yaml_constructors, read_yaml_config

from dataset_classes.dataloader_builders import build_dataloaders
from utils.visualize import create_results_fig
from loops.evaluation import test_loop
from losses.loss_builder import build_loss

def main_evaluation(artifact_name):

    print("-"*80)
    print("Starting SR-IGN evaluation script...")
    print("-"*80)
    print("\n")

    run_name = re.match(r"(.+)\.config_weights", artifact_name).group(1)
    run_name = f"Eval_{run_name}"

    print("-"*80)
    run = wandb.init(
                project = "Super Resolution Using an Idempotent Neural Network",
                name = run_name,
                entity = "porat-hai-technion-israel-institute-of-technology"
                )
    
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()  # downloads locally
    print("-"*80)
    print("\n")

    config_path = f"{artifact_dir}/config.yaml"
    weights_path = f"{artifact_dir}/weights.pt"

    
    print("-"*80)
    register_yaml_constructors()
    print("-"*80)

    config = read_yaml_config(config_path)
    print("-"*80)
    print("\n")

    model_config = config["model"]
    model = Autoencoder(model_config["encoder"], model_config["decoder"])

    
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(constants.device)
    model.eval()

    print("-"*80)
    print("Model architecture:\n")
    print(model)
    print("-"*80)
    print("\n")


    print("-"*80)
    print("Builders:")

    batch_size = config["training"]["batch_size"]
    data_config = config["data"]
    _, test_loader, _ = build_dataloaders(batch_size, **data_config)
    print("----  dataloaders built successfully.\n")

    _, test_loss = build_loss(config)
    print(f"----  loss function built successfully. Test loss: {test_loss.__name__}")
    print("-"*80)
    print("\n\n")
    loss_config = config["loss"]

    print("-"*80)
    print("Test evaluation:")
    test_loop(test_loader, model, test_loss, is_test=True, **loss_config["params"])
    print("Test evaluation completed.")
    print("-"*80)
    print("\n")


    print("-"*80)
    print("Generating visual results...")
    n = 10

    for _ in range(5):
        LR_batch, HR_batch = next(iter(test_loader))
        current_batch_size = LR_batch.size(0)

        n_to_select = min(n, current_batch_size)
        shuffled_indices = torch.randperm(current_batch_size)[:n_to_select]

        LR_batch = LR_batch[shuffled_indices].to(constants.device)
        HR_batch = HR_batch[shuffled_indices].to(constants.device)

        with torch.no_grad():
            SR_batch = model(LR_batch.to(constants.device))

        fig = create_results_fig(LR_batch, SR_batch, HR_batch)

        wandb.log({"Test/Visuale Results": wandb.Image(fig)})

        plt.close(fig)

    print("Visual Results saved successfully.")
    print("-"*80)
    print("\n")

    print("-"*80)
    wandb.finish()
    print("-"*80)
    print("\n")