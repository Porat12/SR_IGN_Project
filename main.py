import torch

import argparse

import constants

from models.model_builder import Autoencoder
from yaml_processing import register_yaml_constructors, read_yaml_config
from optimization_builders import build_optimizer, build_scheduler
from losses.loss_builder import build_loss

from loops.epochs_loop import epochs_loop

from SRdata.dataloader_builders import build_dataloaders

parser = argparse.ArgumentParser()

parser.add_argument('--relative_path_to_config')

args = parser.parse_args()


print("-"*80)
print("Starting SR-IGN training script...")
print(f"Using {constants.device} device.")
print("-"*80)

# ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
# psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

config_path = f"configurations\{args.relative_path_to_config}"

register_yaml_constructors()
print("-"*80)

config = read_yaml_config(config_path)
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
print("-"*80)



print("Builders:")

data_config = config["data"]
train_loader, test_loader = build_dataloaders(batch_size, **data_config)
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


epochs_loop(epochs, train_loader, test_loader, model, model_copy, train_loss, test_loss, optimizer, scheduler, is_batch_scheduler, **loss_config["params"])


# maybe should save in a folder
torch.save(model.state_dict(), 'model_weights.pth')
print("Weights saved successfully.")
print("-"*80)
