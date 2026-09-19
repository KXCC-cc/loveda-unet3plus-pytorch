"""U-Net 3+ 使用的基础卷积块。"""

from torch import nn


class ConvBNReLU(nn.Sequential):
    """一个卷积投影或融合层：Conv2d → BatchNorm2d → ReLU。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    """Encoder 中连续两次 Conv → BN → ReLU，空间尺寸保持不变。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )
