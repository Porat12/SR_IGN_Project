# import matplotlib.pyplot as plt
# import torchvision.utils as vutils

# from utils.metrics import pnsr, ssim

# # convert from [C,H,W] to [H,W,C] for matplotlib
# def show(tensor, title):
#     img = tensor.permute(1, 2, 0).cpu().numpy()
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(title)

# def show_results(LR_images, SR_images, HR_images, save_path):
#     n = len(LR_images)

#     bicubic_result_psnr = [pnsr(LR_images[i], HR_images[i]) for i in range(n)]
#     bicubic_result_ssims = [ssim(LR_images[i], HR_images[i]) for i in range(n)]

#     model_result_psnr = [pnsr(SR_images[i], HR_images[i]) for i in range(n)]
#     model_result_ssims = [ssim(SR_images[i], HR_images[i]) for i in range(n)]


#     lr_grid = vutils.make_grid(LR_images, nrow=n)
#     sr_grid = vutils.make_grid(SR_images, nrow=n)
#     hr_grid = vutils.make_grid(HR_images, nrow=n)

#     plt.figure(figsize=(24, 8))

#     plt.subplot(3, 1, 1)
#     show(lr_grid, "LR batch")

#     plt.subplot(3, 1, 2)
#     show(sr_grid, "SR batch")

#     plt.subplot(3, 1, 3)
#     show(hr_grid, "HR batch")

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()


import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils.metrics import pnsr, ssim

# convert from [C,H,W] to [H,W,C] for matplotlib
def show(tensor, title):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    # assuming the images are normalized to [0, 1]
    img = img.clip(0, 1) 
    
    plt.imshow(img)
    plt.axis("off")
    plt.title(title, fontsize=10)

def show_results(LR_images, SR_images, HR_images, save_path):
    n = len(LR_images)

    bicubic_result_psnr = [pnsr(LR_images[i], HR_images[i]) for i in range(n)]
    bicubic_result_ssims = [ssim(LR_images[i], HR_images[i]) for i in range(n)]

    model_result_psnr = [pnsr(SR_images[i], HR_images[i]) for i in range(n)]
    model_result_ssims = [ssim(SR_images[i], HR_images[i]) for i in range(n)]

    # The layout will be 3 rows (LR, SR, HR) and 'n' columns (for each image).
    plt.figure(figsize=(3 * n, 9)) # Adjust figure size for 'n' images side-by-side

    for i in range(n):
        # --- LR Images (Row 1) ---
        plt.subplot(3, n, i + 1)
        # Format metrics for the title
        title_lr = (
            f"LR\nPSNR: {bicubic_result_psnr[i]:.2f} dB\n"
            f"SSIM: {bicubic_result_ssims[i]:.4f}"
        )
        show(LR_images[i], title_lr)

        # --- SR Images (Row 2) ---
        plt.subplot(3, n, n + i + 1)
        # Format metrics for the title
        title_sr = (
            f"SR\nPSNR: {model_result_psnr[i]:.2f} dB\n"
            f"SSIM: {model_result_ssims[i]:.4f}"
        )
        show(SR_images[i], title_sr)

        # --- HR Images (Row 3) ---
        plt.subplot(3, n, 2 * n + i + 1)
        # HR images are the ground truth, so typically no metrics are reported against themself.
        # We can just use a simple title.
        title_hr = f"HR {i + 1}"
        show(HR_images[i], title_hr)

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()