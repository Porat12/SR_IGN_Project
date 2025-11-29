from enum import Enum

from .blocks import *


class BlockType(Enum):
    DOWN_STRIDE_CONV = DownStrideConvBlock
    UP_STRIDE_CONV = UpStrideConvTransposeBlock

    DOWN_CONV_MAXPOOL = DownConvMaxpoolBlock
    UP_CONV_MAXPOOL = UpConvMaxpoolBlock
