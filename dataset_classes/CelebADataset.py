from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, root_dir, split, image_size, scale):
        self.root_dir = root_dir
        self.image_size = image_size
        self.scale = scale


        self.base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ])

        self.celeba_data = datasets.CelebA(
            root=root_dir,
            split=split,
            download=True,
            transform=self.base_transform
        )
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.celeba_data)

    def __getitem__(self, idx):
        hr_img, _ = self.celeba_data[idx]
        
        w, h = hr_img.size
        
        lr_small = hr_img.resize(
            (w // self.scale, h // self.scale), 
            resample=Image.BICUBIC
        )
        
        lr_restored = lr_small.resize((w, h), resample=Image.BICUBIC)

        return self.to_tensor(lr_restored), self.to_tensor(hr_img)