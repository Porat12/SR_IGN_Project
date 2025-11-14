import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import Conv_Relu_Architecture, Conv_Relu_MaxPool_Architecture
from losses import SR_IGN_loss_for_train, SR_IGN_loss_for_test
import constants

from train import train_loop
from evaluation import test_loop

import time
import math

import argparse
from helpers import get_data_loader


# convert from [C,H,W] to [H,W,C] for matplotlib
def show(tensor, title):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name')
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

# model_args = {"encoder": 
#     [{"in_channels":1,"out_channels":32,"kernel_size":4,"padding":1,"stride":2},
#     {"in_channels":32,"out_channels":128,"kernel_size":4,"padding":1,"stride":2},
#     {"in_channels":128,"out_channels":256,"kernel_size":4,"padding":1,"stride":2},
#     {"in_channels":256,"out_channels":512,"kernel_size":4,"padding":1,"stride":2}],
#     "decoder":
#     [{"in_channels":512,"out_channels":256,"kernel_size":4,"padding":1,"stride":2},
#      {"in_channels":256,"out_channels":128,"kernel_size":4,"padding":1,"stride":2},
#      {"in_channels":128,"out_channels":32,"kernel_size":4,"padding":1,"stride":2},
#      {"in_channels":32,"out_channels":1,"kernel_size":4,"padding":1,"stride":2}]
#     }
    
model_args = {"encoder": 
    [{"in_channels":3,"out_channels":6,"kernel_size":4,"padding":1,"stride":2},
    {"in_channels":6,"out_channels":10,"kernel_size":4,"padding":1,"stride":2}],
    "decoder":
    [{"in_channels":10,"out_channels":6,"kernel_size":4,"padding":1,"stride":2},
     {"in_channels":6,"out_channels":3,"kernel_size":4,"padding":1,"stride":2}]
    }

# model_args = {"encoder": 
#               [{"in_channels":1, "out_channels":8, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2},
#                {"in_channels":8, "out_channels":32, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2},
#                {"in_channels":32, "out_channels":64, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2},
#                {"in_channels":64, "out_channels":128, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2}],
#             "decoder": 
#             [{"in_channels":128, "out_channels":64, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2},
#              {"in_channels":64, "out_channels":32, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2},
#              {"in_channels":32, "out_channels":8, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2},
#              {"in_channels":8, "out_channels":1, "conv_kernel_size":3, "conv_padding":1, "pool_kernel_size":2, "pool_stride":2}]
#             }

model = Conv_Relu_Architecture(model_args).to(constants.device)
model_copy = Conv_Relu_Architecture(model_args).to(constants.device)

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