from torch.utils.data import DataLoader

from .MNISTDataset import MNISTDataset
from .CelebADataset import CelebADataset
from .DIV2KDataset import DIV2KDataset

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
        
        return train_loader, test_loader

    elif data_config["name"] == 'CelebA':
        train_data = CelebADataset(
            root_dir="./data",
            split = 'train',
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

        test_loader = DataLoader(
                        test_data,
                        batch_size = batch_size,
                        shuffle=True
                        )
        
        return train_loader, test_loader
    
    elif data_config["name"] == 'DIV2K':

        train_data = DIV2KDataset(
            root_dir = "data/DIV2K/DIV2K_train_HR/DIV2K_train_HR",
            crop_size = data_config["crop_size"],
            scale_factor = data_config["scale_factor"]
            
        )

        test_data = DIV2KDataset(
            root_dir="data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR",
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
        
        return train_loader, test_loader
    
    return None, None


