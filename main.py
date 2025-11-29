import torch
import wandb
import shutil

import argparse

import constants

import matplotlib.pyplot as plt

from models.model_builder import Autoencoder
from utils.yaml_processing import register_yaml_constructors, read_yaml_config
from optimization_builders import build_optimizer, build_scheduler
from losses.loss_builder import build_loss

from loops.epochs_loop import epochs_loop
from loops.evaluation import test_loop

from dataset_classes.dataloader_builders import build_dataloaders
from utils.visualize import create_results_fig

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--relative_path_to_config')
parser.add_argument('--results_path', default='results.png')

args = parser.parse_args()


config_path = f"configurations/{args.relative_path_to_config}"

print("\n")
print("-"*80)
register_yaml_constructors()
print("-"*80)

config = read_yaml_config(config_path)
run_name = f"{ config["model"]["name"] }_{config["data"]["name"]}"
print("-"*80)
print("\n")

print("-"*80)
if args.train:
    print("Starting SR-IGN training script...")
print(f"Using {constants.device} device.")
print("-"*80)
print("\n")



print("-"*80)
wandb.login(key=constants.wandb_key) # for HPC

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

if args.train:
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

if args.train:
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

    print("logging artifact...")
    torch.save(model.state_dict(), "tmp_artifact_files/weights.pt")
    shutil.copy(config_path, "tmp_artifact_files/config.yaml")

    artifact = wandb.Artifact(
                name=f"{wandb.run.name}_config & weights",
                type="config & weights",
                description="Config, and final weights"
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
print("Generating visual results...")
model.load_state_dict(torch.load('./model_weights.pth', weights_only=True))
model = model.to(constants.device)
model.eval()

n = 10

for i in range(5):
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
