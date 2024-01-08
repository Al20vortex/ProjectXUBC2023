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
        # self.first_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels,
        #               out_channels=out_channels,
        #               kernel_size=kernel_size,
        #               padding="same"),
        #     nn.BatchNorm2d(num_features=out_channels),
        #     nn.ReLU()
        # )
        self.convs = nn.ModuleList(convs)
        # self.additional_layers = nn.ModuleList()
        self.device = get_device()
        # self.first_identity_output = None

    def forward(self, x):
        # x = self.first_conv(x)

        # for i, layer in enumerate(self.additional_layers):
        #     x = layer(x)
        #     if i == 0 and self.first_identity_output is None:  # Store the output of the first IdentityConvLayer once
        #         self.first_identity_output = x

        # if self.count == 3 and self.first_identity_output is not None:
        #     x = x + self.first_identity_output  # Skip connection
        for layer in self.convs:
            x = layer(x)

        return x

    def add_layer(self):
        if self.count < 10:
            new_layer = IdentityConvLayer(
                channels=self.out_channels).to(self.device)
            # self.convs.insert(len(self.convs)-1, new_layer)
            # self.additional_layers.append(new_layer)
            self.convs.append(new_layer)
            self.count += 1


if __name__ == "__main__":
    random_tensor = torch.rand(32, 3, 28, 28)
    block = ConvBlock(in_channels=3, out_channels=4, kernel_size=3)
    print(block)
    block.add_layer()
    print(block)
    print(block(random_tensor).shape)
