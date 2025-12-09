from torchvision.datasets.utils import download_and_extract_archive
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --------------------------
# Settings
# --------------------------
save_root = "/rg/shocher_prj/porat.hai/SR_IGN_Project/data/ImageNet64"
os.makedirs(save_root, exist_ok=True)

# --------------------------
# Function to convert .bin to images (no classes)
# --------------------------
def convert_bin_to_images_noclass(bin_files, output_root):
    os.makedirs(output_root, exist_ok=True)
    counter = 0
    for bin_file in bin_files:
        print(f"Processing {bin_file}...")
        data = np.fromfile(bin_file, dtype=np.uint8)
        n_images = len(data) // (1 + 3*64*64)
        data = data.reshape(n_images, 1 + 3*64*64)
        images = data[:, 1:]  # ignore label
        
        for i in tqdm(range(n_images)):
            img = images[i].reshape(3, 64, 64).transpose(1,2,0)  # HWC
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(output_root, f"{counter}.png"))
            counter += 1
        
        # Remove bin file after conversion
        os.remove(bin_file)

# --------------------------
# Download and extract
# --------------------------
splits = {
    "train": "train_64x64.tar.gz",
    "val": "val_64x64.tar.gz",
    "test": "test_64x64.tar.gz"
}

urls = {
    "train": "https://patrykchrabaszcz.github.io/Imagenet64/train_64x64.tar.gz",
    "val": "https://patrykchrabaszcz.github.io/Imagenet64/val_64x64.tar.gz",
    "test": "https://patrykchrabaszcz.github.io/Imagenet64/test_64x64.tar.gz"
}

for split, filename in splits.items():
    print(f"________________ Downloading ImageNet64 {split.capitalize()} Dataset __________________")
    download_and_extract_archive(
        url=urls[split],
        download_root=save_root,
        extract_root=os.path.join(save_root, split),
        filename=filename
    )

# --------------------------
# Convert binaries to images (no classes)
# --------------------------
for split in ["train", "val", "test"]:
    bin_folder = os.path.join(save_root, split)
    output_folder = os.path.join(save_root, f"{split}_images")
    bin_files = [os.path.join(bin_folder, f) for f in os.listdir(bin_folder) if f.endswith(".bin")]
    
    print(f"________________ Converting {split.capitalize()} Split to PNG (no classes) __________________")
    convert_bin_to_images_noclass(bin_files, output_folder)
    
    # Remove the .tar.gz archive after extraction
    tar_path = os.path.join(save_root, splits[split])
    if os.path.exists(tar_path):
        os.remove(tar_path)

print("________________ Done __________________")
