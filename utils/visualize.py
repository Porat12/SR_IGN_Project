import matplotlib.pyplot as plt
import torchvision.utils as vutils

# convert from [C,H,W] to [H,W,C] for matplotlib
def show(tensor, title):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)

def show_results(LR_images, SR_images, HR_images, save_path):
    n = len(LR_images)

    lr_grid = vutils.make_grid(LR_images, nrow=n)
    sr_grid = vutils.make_grid(SR_images, nrow=n)
    hr_grid = vutils.make_grid(HR_images, nrow=n)

    plt.figure(figsize=(24, 8))

    plt.subplot(3, 1, 1)
    show(lr_grid, "LR batch")

    plt.subplot(3, 1, 2)
    show(sr_grid, "SR batch")

    plt.subplot(3, 1, 3)
    show(hr_grid, "HR batch")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()