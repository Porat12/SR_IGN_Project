import torch
import constants
# from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

@torch.no_grad()
def test_loop(dataloader, model, loss_func, args):
    # Set the model to evaluation mode
    model.eval()

    lam_rec, lam_idem = args.lam_rec, args.lam_idem
    lam_tight, lam_SR = args.lam_tight, args.lam_SR
    a = args.a

    num_batches = len(dataloader)

    loss_history = []
    info_history = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    for LR_img, HR_img in dataloader:
        # Move data to the device
        LR_img = LR_img.to(constants.device)
        HR_img = HR_img.to(constants.device)

        SR_img = model(LR_img)

        loss, info = loss_func(model, LR_img, HR_img, lam_rec, lam_idem, lam_tight, lam_SR, a)
        
        loss_history.append(loss.item())
        info_history.append(info.values())

        # psnr_metric.update(SR_img, HR_img)
        # ssim_metric.update(SR_img, HR_img)

    avg_loss = torch.mean(torch.tensor(loss_history)).item()
    
    # Calculate mean of each entry in info_history
    avg_info = torch.mean(torch.tensor(info_history), dim=0).tolist()
    avg_info = [avg_loss] + avg_info

    # epoch_ssim = ssim_metric.compute()
    # epoch_psnr = psnr_metric.compute()

    # test_ssim = epoch_ssim.item()
    # test_PSNR = epoch_psnr.item()

    # ssim_metric.reset()
    # psnr_metric.reset()

    output = f"Avg Test loss: {avg_loss:>8f}"
    # output += f"| Avg PSNR: {test_PSNR:>4f} dB | Avg SSIM: {test_ssim:>4f}\n"
    print(output)

    return dict(zip(constants.history_keys, avg_info))