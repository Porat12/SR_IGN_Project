from enum import Enum

from .blocks import *


class BlockType(Enum):
    DOWN_CONV_RELU = DownConvReluBlock
    UP_CONV_RELU = UpConvTransposeReluBlock

    DOWN_CONV_RELU_MAXPOOL = DownConvReluMaxpoolBlock
    UP_CONV_RELU_MAXPOOL = UpConvTransposeReluMaxpoolBlock
