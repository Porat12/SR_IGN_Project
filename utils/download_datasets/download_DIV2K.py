from torchvision.datasets.utils import download_and_extract_archive
import os

# save_root = "../../data/DIV2K" # for my local pc
save_root = "/rg/shocher_prj/porat.hai/SR_IGN_Project/data"     # for HPC
os.makedirs(save_root, exist_ok=True)

# Download Train HR
download_and_extract_archive(
    url="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    download_root=save_root,
    extract_root=os.path.join(save_root, "DIV2K_train_HR")
)

# Download Valid HR
download_and_extract_archive(
    url="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    download_root=save_root,
    extract_root=os.path.join(save_root, "DIV2K_valid_HR")
)
