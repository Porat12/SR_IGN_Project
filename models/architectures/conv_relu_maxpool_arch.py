import torch.nn as nn
from typing import List, Dict
from ..encoders import Conv_Relu_MaxPool_Encoder
from ..decoders import Conv_Relu_MaxPool_Decoder

class Conv_Relu_MaxPool_Architecture(nn.Module):
    def __init__(self, architecture_args: Dict[str, List[Dict[str, int]]] ):
        super().__init__()

        self.encoder = Conv_Relu_MaxPool_Encoder(architecture_args["encoder"])
        self.decoder = Conv_Relu_MaxPool_Decoder(architecture_args["decoder"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x