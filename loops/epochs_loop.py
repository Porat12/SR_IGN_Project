import wandb

import time
import math

from .train import train_loop
from .evaluation import test_loop
from utils.wandb_utils import log_with_prefix

def epochs_loop(epochs, train_loader, val_loader, model, model_copy, train_loss, test_loss, optimizer, scheduler, is_batch_scheduler, **loss_params):
    epoch_time = 0.0
    for t in range(epochs):
        print("-------------------------------")
        print(f"Epoch [{t+1}/{epochs}], Avg epoch time {epoch_time:>4f} sec \n-------------------------------")
        epoch_start_time = time.time()

        train_stats = train_loop(train_loader, model, model_copy, train_loss, optimizer, scheduler, is_batch_scheduler, **loss_params)
        val_stats = test_loop(val_loader, model, test_loss, is_test=False, **loss_params)

        wandb.log({
                    **log_with_prefix("train", train_stats),
                    **log_with_prefix("validation", val_stats),
                    "epoch": t+1,
                })

        epoch_end_time = time.time()
        epoch_time = (epoch_time * t + epoch_end_time - epoch_start_time) / (t + 1)

    
    print(f"Total time {math.floor(epoch_time * epochs)} sec \n-------------------------------")
    print("Done!")