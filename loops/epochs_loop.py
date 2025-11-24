import time
import math

from.train import train_loop
from.evaluation import test_loop


def epochs_loop(epochs, train_loader, test_loader, model, model_copy, train_loss, test_loss, optimizer, scheduler, is_batch_scheduler, **loss_params):
    epoch_time = 0.0
    for t in range(epochs):
        print(f"Epoch [{t+1}/{epochs}], Avg epoch time {epoch_time:>4f} sec \n-------------------------------")
        epoch_start_time = time.time()

        train_info = train_loop(train_loader, model, model_copy, train_loss, optimizer, scheduler, is_batch_scheduler, **loss_params)
        test_info = test_loop(test_loader, model, test_loss, **loss_params)


        epoch_end_time = time.time()
        epoch_time = (epoch_time * t + epoch_end_time - epoch_start_time) / (t + 1)

    print(f"Total time {math.floor(epoch_time * epochs)} sec \n-------------------------------")
    print("Done!")