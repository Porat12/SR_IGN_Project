import torch.nn as nn

class DownConvMaxpoolBlock(nn.Module):
    
    def __init__(self, activation, in_channels, out_channels, conv_kernel_size, conv_padding, pool_kernel_size, pool_stride):
        super().__init__()
        ActivationClass = getattr(nn, activation)

        self.block = nn.Sequential(
            nn.Conv2d( in_channels = in_channels,  out_channels = out_channels, 
                       kernel_size = conv_kernel_size, padding = conv_padding, stride=1 ),
            ActivationClass(),
            nn.MaxPool2d(kernel_size = pool_kernel_size, stride = pool_stride)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class UpConvMaxpoolBlock(nn.Module):
    
    def __init__(self, activation, in_channels, out_channels, conv_kernel_size, conv_padding, scale_factor):
        super().__init__()
        ActivationClass = getattr(nn, activation)
        self.block = nn.Sequential(
            nn.Upsample(scale_factor = scale_factor, mode = 'nearest'),
            nn.Conv2d( in_channels = in_channels, out_channels = out_channels,
                       kernel_size = conv_kernel_size, padding=conv_padding, stride=1),
            ActivationClass(),
        )
        
    def forward(self, x):
        x = self.block(x)
        return x
