import matplotlib.pyplot as plt
from utils.metrics import pnsr, ssim

# convert from [C,H,W] to [H,W,C] for matplotlib
def show(tensor, title = ""):
    img = tensor.permute(1, 2, 0).cpu().numpy()
        
    plt.imshow(img)
    plt.axis("off")
    plt.title(title, fontsize=10)

def show_results(LR_images, SR_images, HR_images, save_path):

    n = len(LR_images)

    fig = plt.figure(figsize=(3 * n, 9)) 

    for i in range(n):
        # --- LR Images (Row 1) ---
        plt.subplot(3, n, i + 1)
        show(LR_images[i])

        # --- SR Images (Row 2) ---
        plt.subplot(3, n, n + i + 1)
        show(SR_images[i])

        # --- HR Images (Row 3) ---
        plt.subplot(3, n, 2 * n + i + 1)
        show(HR_images[i])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    

def create_results_fig(LR_images, SR_images, HR_images):
    n = len(LR_images)

    bicubic_result_psnr = [pnsr(LR_images[i], HR_images[i]) for i in range(n)]
    bicubic_result_ssims = [ssim(LR_images[i], HR_images[i]) for i in range(n)]

    model_result_psnr = [pnsr(SR_images[i], HR_images[i]) for i in range(n)]
    model_result_ssims = [ssim(SR_images[i], HR_images[i]) for i in range(n)]

    # Create the figure object explicitly
    fig = plt.figure(figsize=(3 * n, 9)) 

    for i in range(n):
        # --- LR Images (Row 1) ---
        plt.subplot(3, n, i + 1)
        title_lr = (
            f"LR\nPSNR: {bicubic_result_psnr[i]:.2f} dB\n"
            f"SSIM: {bicubic_result_ssims[i]:.4f}"
        )
        show(LR_images[i], title_lr)

        # --- SR Images (Row 2) ---
        plt.subplot(3, n, n + i + 1)
        title_sr = (
            f"SR\nPSNR: {model_result_psnr[i]:.2f} dB\n"
            f"SSIM: {model_result_ssims[i]:.4f}"
        )
        show(SR_images[i], title_sr)

        # --- HR Images (Row 3) ---
        plt.subplot(3, n, 2 * n + i + 1)
        title_hr = f"HR {i + 1}"
        show(HR_images[i], title_hr)

    plt.tight_layout()
    
    return fig