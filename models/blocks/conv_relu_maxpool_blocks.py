import torch.nn as nn

class DownConvReluMaxpoolBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_padding, pool_kernel_size, pool_stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d( in_channels = in_channels,  out_channels = out_channels, 
                       kernel_size = conv_kernel_size, padding = conv_padding, stride=1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_kernel_size, stride = pool_stride)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class UpConvReluMaxpoolBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_padding, scale_factor):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor = scale_factor, mode = 'nearest'),
            nn.Conv2d( in_channels = in_channels, out_channels = out_channels,
                       kernel_size = conv_kernel_size, padding=conv_padding, stride=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.block(x)
        return x
