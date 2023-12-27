import torch
import torch.nn as nn
from .identity_conv import IdentityConvLayer
from .utils import get_device


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int) -> None:
        super().__init__()
        convs = list()
        self.count = 0
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        convs.extend(
            [nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding="same"),
             nn.BatchNorm2d(num_features=out_channels),
             nn.ReLU()]
        )
        self.convs = nn.ModuleList(convs)
        self.device = get_device()

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x

    def add_layer(self):
        if self.count < 3:
            new_layer = IdentityConvLayer(
                channels=self.out_channels).to(self.device)
            self.convs.insert(len(self.convs)-1, new_layer)
            self.count += 1


if __name__ == "__main__":
    random_tensor = torch.rand(32, 3, 28, 28)
    block = ConvBlock(in_channels=3, out_channels=4, kernel_size=3)
    print(block)
    block.add_layer()
    print(block)
    print(block(random_tensor).shape)
