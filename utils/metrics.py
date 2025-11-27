import torch
from torchvision.transforms.functional import gaussian_blur

def psnr_batch(img_batch1, img_batch2, max_pixel=1.0):

    mse = torch.mean((img_batch1 - img_batch2) ** 2, dim=[1, 2, 3])
    psnr = torch.log10((max_pixel ** 2) / mse)

    return 10 * psnr.mean().item()

def pnsr(img1, img2, max_pixel=1.0):
    if img1.dim() != 3 or img2.dim() != 3:
        raise ValueError("Input images must be 3-dimensional tensors (C, H, W).")
    
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    return psnr_batch(img1, img2, max_pixel)

def ssim_batch(x, y, window_size=11, sigma=1.5, max_val=1.0):    
    B, C, H, W = x.shape

    mu_x = gaussian_blur(x, kernel_size=(window_size, window_size), sigma=(sigma, sigma))
    mu_y = gaussian_blur(y, kernel_size=(window_size, window_size), sigma=(sigma, sigma))

    sigma_x2 = gaussian_blur(x * x, kernel_size=(window_size, window_size), sigma=(sigma, sigma)) - mu_x**2
    sigma_y2 = gaussian_blur(y * y, kernel_size=(window_size, window_size), sigma=(sigma, sigma)) - mu_y**2
    sigma_xy = gaussian_blur(x * y, kernel_size=(window_size, window_size), sigma=(sigma, sigma)) - mu_x * mu_y

    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den

    return ssim_map.mean(dim=[1,2,3]).mean().item()

def ssim(x, y, window_size=11, sigma=1.5, max_val=1.0):
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("Input images must be 3-dimensional tensors (C, H, W).")
    
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    
    return ssim_batch(x, y, window_size, sigma, max_val)
