from torch.utils.data import DataLoader

from .MNISTDataset import MNISTDataset
from .CelebADataset import CelebADataset
from .DIV2KDataset import DIV2KDataset
from .ImageNet64Dataset import ImageNet64Dataset

def build_dataloaders(batch_size, **data_config):

    if data_config["name"] == 'MNIST':

        train_data = MNISTDataset(
            root_dir="./data",
            image_size = data_config["img_size"],
            scale = data_config["scale_factor"],
            is_train = True
        )

        test_data = MNISTDataset(
            root_dir="./data",
            image_size = data_config["img_size"],
            scale = data_config["scale_factor"],
            is_train = False
        )

        train_loader = DataLoader(
                        train_data,
                        batch_size = batch_size,
                        shuffle=True
                        )

        test_loader = DataLoader(
                        test_data,
                        batch_size = batch_size,
                        shuffle=True
                        )
        
        return train_loader, test_loader, None

    elif data_config["name"] == 'CelebA':
        train_data = CelebADataset(
            root_dir="./data",
            split = 'train',
            image_size = data_config["img_size"],
            scale = data_config["scale_factor"]
            
        )

        valid_data = CelebADataset(
            root_dir="./data",
            split = 'valid',
            image_size = data_config["img_size"],
            scale = data_config["scale_factor"]
            
        )

        test_data = CelebADataset(
            root_dir="./data",
            split = 'test',
            image_size = data_config["img_size"],
            scale = data_config["scale_factor"]
            
        )

        train_loader = DataLoader(
                        train_data,
                        batch_size = batch_size,
                        shuffle=True
                        )

        valid_loader = DataLoader(
                        valid_data,
                        batch_size = batch_size,
                        shuffle=True
                        )
        
        test_loader = DataLoader(
                        test_data,
                        batch_size = batch_size,
                        shuffle=True
                        )
        
        return train_loader, test_loader, valid_loader
    
    elif data_config["name"] == 'DIV2K':

        train_data = DIV2KDataset(
            root_dir = "data/DIV2K/DIV2K_train_HR/DIV2K_train_HR",
            crop_size = data_config["crop_size"],
            scale_factor = data_config["scale_factor"]
            
        )

        test_data = DIV2KDataset(
            root_dir = "data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR",
            crop_size = data_config["crop_size"],
            scale_factor = data_config["scale_factor"]
            
        )

        train_loader = DataLoader(
                        train_data,
                        batch_size = batch_size,
                        shuffle=True
                        )

        test_loader = DataLoader(
                        test_data,
                        batch_size = batch_size,
                        shuffle=True
                        )
        
        return train_loader, test_loader, None
    
    elif data_config["name"] == 'ImageNet64':

        train_data = ImageNet64Dataset(
            root_dir = "data/ImageNet64/train_64x64",
            split = "train",
            scale_factor = data_config["scale_factor"]
            
        )

        test_data = ImageNet64Dataset(
            root_dir = "data/ImageNet64/valid_64x64",
            split = "valid",
            scale_factor = data_config["scale_factor"]
        )

        train_loader = DataLoader(
                        train_data,
                        batch_size = batch_size,
                        shuffle=True
                        )

        test_loader = DataLoader(
                        test_data,
                        batch_size = batch_size,
                        shuffle=True
                        )
        
        return train_loader, test_loader, None
    
    return None, None, None


