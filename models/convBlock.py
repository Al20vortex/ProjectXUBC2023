import torch
import torch.nn as nn
from utils import get_device
from .identityConv import IdentityConvLayer


class ConvBlock(nn.Module):
    """
    An expandable block. 
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            pooling_amount: int,
            dropout: float) -> None:
        super().__init__()
        convs = list()
        self.count = 0
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        convs.extend(
            [nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding="same"),
             nn.Dropout2d(self.dropout),
             nn.BatchNorm2d(num_features=out_channels),
             nn.LeakyReLU(0.2),
             nn.MaxPool2d(pooling_amount)
             ]
        )
        self.convs = nn.ModuleList(convs)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x)
        return x

    def add_layer(self):
        """
        Adds an identity layer if the maximum capacity is not yet filled
        """
        if self.count < 10:  # BEST 5
            new_layer = IdentityConvLayer(
                channels=self.out_channels).to(self.device)
            self.convs.insert(len(self.convs)-1, new_layer)
            self.count += 1
