import torch.nn as nn
from typing import List, Dict
from ..blocks.conv_relu_maxpool_blocks import Down_Conv_Relu_MaxPool_Block

class Conv_Relu_MaxPool_Encoder(nn.Module):
    
    def __init__(self, block_args_list: List[Dict[str, int]]):
        super().__init__()
        
        # Create a list of blocks
        blocks = [Down_Conv_Relu_MaxPool_Block(**block_args) for block_args in block_args_list]
        
        # Stack all blocks in a sequential
        self.encoder = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.encoder(x)
        return x