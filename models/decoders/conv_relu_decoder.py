import torch.nn as nn
from typing import List, Dict
from ..blocks.conv_relu_blocks import Up_ConvT_Relu_Block

class Conv_Relu_Decoder(nn.Module):
    
    def __init__(self, block_args_list: List[Dict[str, int]]):
        super().__init__()
        
        # Create a list of blocks
        blocks = [Up_ConvT_Relu_Block(**block_args) for block_args in block_args_list]
        
        # Stack all blocks in a sequential
        self.decoder = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.decoder(x)
        return x