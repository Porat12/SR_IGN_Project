import torch
import wandb

import argparse

import constants

from models.model_builder import Autoencoder
from utils.yaml_processing import register_yaml_constructors, read_yaml_config
from optimization_builders import build_optimizer, build_scheduler
from losses.loss_builder import build_loss

from loops.epochs_loop import epochs_loop

from dataset_classes.dataloader_builders import build_dataloaders
from utils.visualize import show_results

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--relative_path_to_config')
parser.add_argument('--results_path', default='results.png')

args = parser.parse_args()




# ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
# psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

config_path = f"configurations/{args.relative_path_to_config}"

register_yaml_constructors()
print("-"*80)

config = read_yaml_config(config_path)
print("-"*80)


print("-"*80)
if args.train:
    print("Starting SR-IGN training script...")
print(f"Using {constants.device} device.")
print("-"*80)


wandb.login(key=constants.wandb_key) # for HPC

wandb.init(
project = "Super Resolution Using an Idempotent Neural Network",
entity = "porat-hai-technion-israel-institute-of-technology",
config = config
)

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
    print("-"*80)



print("Builders:")

data_config = config["data"]
train_loader, test_loader, valid_loader = build_dataloaders(batch_size, **data_config)
eval_loader = test_loader if valid_loader is None else valid_loader

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


    epochs_loop(epochs, train_loader, eval_loader, valid_loader, model, model_copy, train_loss, test_loss, optimizer, scheduler, is_batch_scheduler, **loss_config["params"])


    # maybe should save in a folder
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Weights saved successfully.")
    print("-"*80)


model.load_state_dict(torch.load('./model_weights.pth', weights_only=True))
model = model.to(constants.device)
model.eval()

n = 5

LR_batch, HR_batch = next(iter(test_loader))
LR_batch = LR_batch[:n]
HR_batch = HR_batch[:n]

with torch.no_grad():
    SR_batch = model(LR_batch.to(constants.device))


show_results(LR_batch, SR_batch, HR_batch, save_path="results.png")

wandb.log({
    "LR": [wandb.Image(img, caption="LR Images") 
                   for img in LR_batch.cpu()],
    "SR": [wandb.Image(img, caption="SR Images") 
                    for img in SR_batch.cpu()],
    "HR": [wandb.Image(img, caption="HR Images") 
                   for img in HR_batch.cpu()]
})

wandb.finish()
