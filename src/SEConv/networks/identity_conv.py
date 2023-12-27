import torch.nn as nn
import torch


class IdentityConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(channels, channels,
                         kernel_size=3, padding="same", bias=False)
        identity_matrix = torch.eye(channels).view(channels, channels, 1, 1)
        print("Identity matrix type: ", identity_matrix.type())
        with torch.no_grad():
            conv.weight.copy_(identity_matrix)
        self.conv = nn.Sequential(conv,
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU())

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)


if __name__ == "__main__":
    model = IdentityConvLayer(3)
    tensor = torch.rand(32, 3, 28, 28)
    print(model)
    print(model(tensor).shape)
