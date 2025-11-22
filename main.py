import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.model_builder import Autoencoder
from yaml_processing import register_yaml_constructors, read_yaml_config
from losses import SR_IGN_loss_for_train, SR_IGN_loss_for_test
import constants

from train import train_loop
from evaluation import test_loop

import time
import math

import argparse
from helpers import get_data_loader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name')
parser.add_argument('--model_cfg_name')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--HR_img_size', type=int)
parser.add_argument('--scale_factor', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--beta_1', type=float)
parser.add_argument('--beta_2', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--lam_rec', type=float)
parser.add_argument('--lam_idem', type=float)
parser.add_argument('--lam_tight', type=float)
parser.add_argument('--lam_SR', type=float)
parser.add_argument('--a', type=float)
parser.add_argument('--epochs', type=int)

args = parser.parse_args()


train_loader, test_loader = get_data_loader(args)



print(f"Using {constants.device} device")

# ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
# psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

model_config_path = f"model_configurations\model_cfg_{args.dataset_name}\{args.model_cfg_name}.yaml"

register_yaml_constructors()
model_config = read_yaml_config(model_config_path)["model"]

model = Autoencoder(model_config["encoder"], model_config["decoder"]).to(constants.device)
model_copy = Autoencoder(model_config["encoder"], model_config["decoder"]).to(constants.device)

print(model)

optimizer = optim.Adam(
    model.parameters(),    # the parameters to optimize
    lr = args.learning_rate,
    betas = (args.beta_1, args.beta_1),    # running averages of grad and grad^2
    eps = 1e-8,
    weight_decay = args.weight_decay        # L2 regularization (optional)
)


scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=10,
#     T_mult=10,
#     eta_min=1e-6
# )


def epochs_loop(epochs):
    epoch_time = 0.0
    for t in range(epochs):
        print(f"Epoch {t+1}, Avg epoch time {epoch_time:>4f} sec \n-------------------------------")
        epoch_start_time = time.time()

        train_info = train_loop(train_loader, model, model_copy, SR_IGN_loss_for_train, args, optimizer, scheduler)
        test_info = test_loop(test_loader, model, SR_IGN_loss_for_test, args)


        epoch_end_time = time.time()
        epoch_time = (epoch_time * t + epoch_end_time - epoch_start_time) / (t + 1)

    print(f"Total time {math.floor(epoch_time * epochs)} sec \n-------------------------------")
    print("Done!")

    # maybe should save in a folder
    torch.save(model.state_dict(), 'model_weights.pth')




epochs_loop(args.epochs)