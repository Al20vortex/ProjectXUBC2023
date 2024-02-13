import torch.nn as nn
import torch
from utils import get_device
NOISE_COEFF = 1e-4
device = get_device()


class IdentityConvLayer(nn.Module):
    """
    An identity conv layer with weights initialized to Identity, with a bit of gausian noise added. 
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(channels, channels,
                         kernel_size=3, padding="same", bias=False)

        # Creating an identity matrix with added noise
        identity_matrix = torch.eye(channels).view(channels, channels, 1, 1)
        noise = torch.randn(identity_matrix.shape) * NOISE_COEFF
        identity_matrix_with_noise = identity_matrix + noise
        with torch.no_grad():
            conv.weight.copy_(identity_matrix_with_noise)
        self.conv = nn.Sequential(conv,
                                  nn.BatchNorm2d(channels),
                                  nn.LeakyReLU(0.2)).to(device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)
