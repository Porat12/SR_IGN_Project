import torch.nn as nn

class DownConvReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d( in_channels = in_channels,  out_channels = out_channels, 
                       kernel_size = kernel_size, padding = padding, stride = stride ),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class UpConvTransposeReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d( in_channels = in_channels, out_channels = out_channels, 
                                kernel_size = kernel_size, padding = padding, stride = stride ),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x