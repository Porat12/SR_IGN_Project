from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F


class SR_celebA(Dataset):
    def __init__(self, root_dir, split, image_size, scale):
        self.root_dir = root_dir
        self.image_size = image_size
        self.scale = scale
        self.HR_data = datasets.CelebA(
              root=self.root_dir,
              split=split,
              download=True,
              transform=transforms.Compose([
                        transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor()#,
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
              )

    def __len__(self):
      return len(self.HR_data)

    def __getitem__(self, idx):
        HR_img, _ = self.HR_data[idx]
        h, w = HR_img.shape[1:]

        # TODO// what if the scale is not perfect? ----------------------------
        #  if after the up sample the dimentions don't match-------------------
        # ---------------------------------------------------------------------
        LR_small = F.interpolate(
                        HR_img.unsqueeze(0),
                        size=(h // self.scale, w // self.scale),
                        mode='bicubic',
                        align_corners=False,
                        **({"antialias": True} if "antialias" in F.interpolate.__code__.co_varnames else {})
                        )

        LR_img = F.interpolate(
                        LR_small,
                        size=(h, w),
                        mode='bicubic',
                        align_corners=False
                        ).squeeze(0)

        return LR_img, HR_img