import torch.nn as nn
from typing import List, Dict
from ..blocks.conv_relu_maxpool_blocks import Up_MaxPool_Relu_ConvT_Block

class Conv_Relu_MaxPool_Decoder(nn.Module):
    
    def __init__(self, block_args_list: List[Dict[str, int]]):
        super().__init__()
        
        # Create a list of blocks and stack them in sequential
        blocks = [Up_MaxPool_Relu_ConvT_Block(**block_args) for block_args in block_args_list]
        self.decoder = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.decoder(x)
        return x