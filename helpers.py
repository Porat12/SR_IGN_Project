from torch.utils.data import DataLoader

from SRdata.SR_MNIST import SR_MNIST
from SRdata.SR_celebA import SR_celebA


def get_data_loader(args):

    if args.dataset_name == 'MNIST':

        train_data = SR_MNIST(
            root_dir="./data",
            image_size = args.HR_img_size,      # HR image size
            scale = args.scale_factor,
            is_train = True
        )

        test_data = SR_MNIST(
            root_dir="./data",
            image_size = args.HR_img_size,      # HR image size
            scale = args.scale_factor,
            is_train = False
        )

        train_loader = DataLoader(
                        train_data,
                        batch_size = args.batch_size,
                        shuffle=True
                        )

        test_loader = DataLoader(
                        test_data,
                        batch_size = args.batch_size,
                        shuffle=True
                        )
        
        return train_loader, test_loader

    elif args.dataset_name == 'celebA':
        train_data = SR_celebA(
            root_dir="./data",
            split = 'train',
            image_size = args.HR_img_size,      # HR image size
            scale = args.scale_factor
            
        )

        test_data = SR_celebA(
            root_dir="./data",
            split = 'test',
            image_size = args.HR_img_size,      # HR image size
            scale = args.scale_factor
            
        )

        train_loader = DataLoader(
                        train_data,
                        batch_size = args.batch_size,
                        shuffle=True
                        )

        test_loader = DataLoader(
                        test_data,
                        batch_size = args.batch_size,
                        shuffle=True
                        )
        return train_loader, test_loader
    return None, None


