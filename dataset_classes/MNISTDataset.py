from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

class MNISTDataset(Dataset):
    def __init__(self, root_dir, image_size, scale, is_train=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.scale = scale
        
        self.base_transform = transforms.Resize((image_size, image_size))
        
        self.mnist_data = datasets.MNIST(
            root=root_dir,
            train=is_train,
            download=True,
            transform=self.base_transform
        )
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        hr_img, _ = self.mnist_data[idx]
        
        w, h = hr_img.size
        
        lr_small = hr_img.resize(
            (w // self.scale, h // self.scale), 
            resample=Image.BICUBIC
        )
        
        lr_restored = lr_small.resize((w, h), resample=Image.BICUBIC)

        return self.to_tensor(lr_restored), self.to_tensor(hr_img)