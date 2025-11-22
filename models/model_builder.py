import torch.nn as nn

from models.block_type import BlockType

def build_block(block_type: BlockType, **block_params):
    block_class  = block_type.value
    return block_class(**block_params)

class ModelBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        for block in config:
            block_type = block['type']
            params = block['params']
            layers.append(build_block(block_type, **params))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ModelBuilder(config)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = ModelBuilder(config)

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, enc_config, dec_config):
        super().__init__()
        self.encoder = Encoder(enc_config)
        self.decoder = Decoder(dec_config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

