import torch.nn as nn

class DownStrideConvBlock(nn.Module):
    def __init__(self, activation, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        ActivationClass = getattr(nn, activation)
        self.block = nn.Sequential(
            nn.Conv2d( in_channels = in_channels,  out_channels = out_channels, 
                       kernel_size = kernel_size, padding = padding, stride = stride ),
            ActivationClass()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class UpStrideConvTransposeBlock(nn.Module):
    def __init__(self, activation, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        ActivationClass = getattr(nn, activation)
        self.block = nn.Sequential(
            nn.ConvTranspose2d( in_channels = in_channels, out_channels = out_channels, 
                                kernel_size = kernel_size, padding = padding, stride = stride ),
            ActivationClass()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x