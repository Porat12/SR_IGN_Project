from torch.utils.data import DataLoader

from .SR_MNIST import SR_MNIST
from .SR_celebA import SR_celebA


def build_dataloaders(batch_size, **data_config):

    if data_config["name"] == 'MNIST':

        train_data = SR_MNIST(
            root_dir="./data",
            image_size = data_config["img_size"],      # HR image size
            scale = data_config["scale_factor"],
            is_train = True
        )

        test_data = SR_MNIST(
            root_dir="./data",
            image_size = data_config["img_size"],      # HR image size
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

    elif data_config["name"] == 'celebA':
        train_data = SR_celebA(
            root_dir="./data",
            split = 'train',
            image_size = data_config["img_size"],      # HR image size
            scale = data_config["scale_factor"]
            
        )

        test_data = SR_celebA(
            root_dir="./data",
            split = 'test',
            image_size = data_config["img_size"],      # HR image size
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
    return None, None


