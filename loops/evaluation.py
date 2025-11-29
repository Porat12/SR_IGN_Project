import torch
import constants
from utils.metrics import psnr_batch, ssim_batch

@torch.no_grad()
def test_loop(dataloader, model, loss_func, is_test, **loss_params):
    # Set the model to evaluation mode
    model.eval()

    num_batches = len(dataloader)

    loss_history = []
    info_history = []

    psnr = 0.0
    ssim = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    for LR_img, HR_img in dataloader:
        # Move data to the device
        LR_img = LR_img.to(constants.device)
        HR_img = HR_img.to(constants.device)

        SR_img = model(LR_img)

        loss, losses_info = loss_func(model, LR_img, HR_img, **loss_params)
        
        loss_history.append(loss.item())
        
        # Enforce a fixed order matching constants.losses_keys
        info_history.append([losses_info[k] for k in constants.losses_keys])

        psnr += psnr_batch(SR_img, HR_img)
        ssim += ssim_batch(SR_img, HR_img)


    avg_loss = torch.mean(torch.tensor(loss_history)).item()
    
    # Calculate mean of each entry in info_history
    avg_info = torch.mean(torch.tensor(info_history, dtype=torch.float32), dim=0).tolist()

    psnr /= num_batches
    ssim /= num_batches

    avg_info = [avg_loss] + avg_info + [psnr, ssim]

    temp = "Test" if is_test else "Validation" 
    output = f"Avg {temp} loss: {avg_loss:>8f}"
    output += f"| Avg PSNR: {psnr:>4f} dB | Avg SSIM: {ssim:>4f}\n"
    print(output)

    return dict(zip(constants.val_history_keys, avg_info))