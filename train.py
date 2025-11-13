import torch

import time
import constants
import math


def train_loop(dataloader, model, model_copy, loss_func, args, optimizer, scheduler = None):
    # Set the model to training mode
    model.train()

    ordered_keys = constants.history_keys[1:]
    lam_rec, lam_idem = args.lam_rec, args.lam_idem
    lam_tight, lam_SR = args.lam_tight, args.lam_SR
    a = args.a

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    report_time = num_batches // 10

    batch_time = 0.0
    loss_history = []
    info_history = []
    for batch_idx, (LR_img, HR_img) in enumerate(dataloader):

        start_time = time.time()

        batch_len = len(LR_img)
        model_copy.load_state_dict(model.state_dict())

        # Move data to the device
        LR_img = LR_img.to(constants.device)
        HR_img = HR_img.to(constants.device)

        # Compute loss
        loss, info = loss_func(model, model_copy, LR_img, HR_img, lam_rec, lam_idem, lam_tight, lam_SR, a)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        

        end_time = time.time()

        batch_time += (end_time - start_time)
        if batch_idx % report_time == 0:
            
            loss_history.append(loss.item())
            
            # Enforce a fixed order matching constants.history_keys (excluding total_loss)
            info_history.append([info[k] for k in ordered_keys])
            
            batch_time /= 50
            current = batch_idx * dataloader.batch_size + batch_len
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]------- Avg batch time: {math.floor(batch_time*10**6)} μs")
            batch_time = 0.0

    avg_loss = torch.mean(torch.tensor(loss_history)).item()
    
    # Calculate mean of each entry in info_history
    avg_info = torch.mean(torch.tensor(info_history, dtype=torch.float32), dim=0).tolist()
    avg_info = [avg_loss] + avg_info

    return dict(zip(constants.history_keys, avg_info))