import tensorflow_datasets as tfds
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import shutil

# --------------------------
# Settings
# --------------------------
save_root = "/rg/shocher_prj/porat.hai/SR_IGN_Project/data/ImageNet64_tfds"
os.makedirs(save_root, exist_ok=True)

# Desired split ratio for test (from validation set)
test_ratio = 0.5  # 50% of validation becomes test

# --------------------------
# Helper functions
# --------------------------
def save_images(ds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, example in enumerate(tfds.as_numpy(ds)):
        img = example["image"]  # uint8, shape (64,64,3)
        Image.fromarray(img).save(os.path.join(out_dir, f"{i}.png"))
        if i % 10000 == 0:
            print(f"Saved {i} images to {out_dir}")
    print(f"Finished saving images to {out_dir}")

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".png")])

# --------------------------
# Download TFDS dataset
# --------------------------
print("Downloading TFDS downsampled_imagenet/64x64 ...")
train_ds = tfds.load("downsampled_imagenet/64x64", split="train", as_supervised=False)
val_ds   = tfds.load("downsampled_imagenet/64x64", split="validation", as_supervised=False)

# --------------------------
# Save train images
# --------------------------
train_dir = os.path.join(save_root, "train_images")
print("Saving train images ...")
save_images(train_ds, train_dir)

# --------------------------
# Split validation into validation + test
# --------------------------
val_list = list(tfds.as_numpy(val_ds))
random.shuffle(val_list)
split_idx = int(len(val_list) * test_ratio)

test_list = val_list[:split_idx]
val_list = val_list[split_idx:]

val_dir = os.path.join(save_root, "validation_images")
test_dir = os.path.join(save_root, "test_images")

print("Saving validation images ...")
os.makedirs(val_dir, exist_ok=True)
for i, example in enumerate(val_list):
    Image.fromarray(example["image"]).save(os.path.join(val_dir, f"{i}.png"))

print("Saving test images ...")
os.makedirs(test_dir, exist_ok=True)
for i, example in enumerate(test_list):
    Image.fromarray(example["image"]).save(os.path.join(test_dir, f"{i}.png"))

# --------------------------
# Sanity checks
# --------------------------
train_count = count_images(train_dir)
val_count   = count_images(val_dir)
test_count  = count_images(test_dir)

print(f"Train images: {train_count}")
print(f"Validation images: {val_count}")
print(f"Test images: {test_count}")

assert train_count > 1_200_000, f"Train set seems incomplete ({train_count} images)"
assert val_count >= 25_000, f"Validation set seems small ({val_count} images)"
assert test_count >= 25_000, f"Test set seems small ({test_count} images)"

print("ImageNet64 dataset download and preparation complete.")

# # --------------------------
# # PyTorch Dataset
# # --------------------------
# class ImageOnlyDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")]
#         self.transform = transform

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         img = Image.open(self.files[idx]).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img

# # --------------------------
# # Example usage
# # --------------------------
# transform = transforms.ToTensor()
# train_dataset = ImageOnlyDataset(train_dir, transform=transform)
# val_dataset   = ImageOnlyDataset(val_dir, transform=transform)
# test_dataset  = ImageOnlyDataset(test_dir, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Test: load a batch
# images = next(iter(train_loader))
# print(images.shape)  # [64, 3, 64, 64]
