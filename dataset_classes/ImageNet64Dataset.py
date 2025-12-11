import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class ImageNet64Dataset(Dataset):
    def __init__(self, root_dir, split, scale_factor):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()
        
        # Define a cache file path inside the root dir
        parent_dir = os.path.dirname(root_dir.rstrip(os.sep))
        cache_file = os.path.join(parent_dir, f"imagenet_{split}_cache.pt")

        if os.path.exists(cache_file):
            print(f"Loading cached file list from {cache_file}...")
            # Load instantly
            self.image_paths = torch.load(cache_file)
        else:
            print(f"Scanning files in {root_dir}... (This may take a while)")
            # The slow part (run only once)
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
            
            # Save for next time
            print(f"Saving cache to {cache_file}...")
            torch.save(self.image_paths, cache_file)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.image_paths[idx]).convert("RGB")
                
        w, h = hr_img.size
        
        lr_small = hr_img.resize(
            (w // self.scale_factor, h // self.scale_factor), 
            resample=Image.BICUBIC
        )
        
        lr_restored = lr_small.resize((w, h), resample=Image.BICUBIC)

        return self.to_tensor(lr_restored), self.to_tensor(hr_img)