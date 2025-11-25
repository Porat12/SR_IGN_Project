import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, crop_size, scale_factor):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        
        self.cropper = transforms.RandomCrop(crop_size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.image_paths[idx]).convert("RGB")
        
        hr_crop = self.cropper(hr_img)
        
        w, h = hr_crop.size
        
        lr_small = hr_crop.resize(
            (w // self.scale_factor, h // self.scale_factor), 
            resample=Image.BICUBIC
        )
        
        lr_restored = lr_small.resize((w, h), resample=Image.BICUBIC)

        return self.to_tensor(lr_restored), self.to_tensor(hr_crop)