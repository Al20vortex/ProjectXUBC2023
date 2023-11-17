import torch
import torch.nn as nn
from Attention import MultiHeadedAttention

class ViT(nn.Module):
    """
    A standard vision transformer for classification
    """
    def __init__(self, in_channels: int=3, out_channels: int=4):
        """
        Args:
            in_channels:
            out_channels:
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, stride=16, kernel_size=16)
        self.dim = 16
        layer_1 = MultiHeadedAttention(dim=16, num_heads=8)
        layer_2 = MultiHeadedAttention(dim=16, num_heads=8)
        layer_3 = MultiHeadedAttention(dim=16, num_heads=8)
        layer_4 = MultiHeadedAttention(dim=16, num_heads=8)

        self.attention = nn.Sequential(
            layer_1,
            layer_2,
            layer_3,
            layer_4
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass
        Args:
            X: the input tensor

        Returns: the output tensor duh

        """
        X = self.conv(X)
        print(f"X shape: {X.shape}")
        outer_dim = X.shape[0]
        X = X.view(outer_dim, -1)
        print(f"X shape again{X.shape}")
        dim = X.shape[1]
        X = MultiHeadedAttention(dim=dim)(X)
        X = MultiHeadedAttention(dim=dim)(X)
        X = MultiHeadedAttention(dim=dim)(X)
        X = MultiHeadedAttention(dim=dim)(X)
        return X


if __name__ == "__main__":
    vit = ViT(in_channels=3, out_channels=3)
    rand = torch.ones(3, 64, 64)
    pred = vit(rand)
    outer_dim = pred.shape[0]
    print(pred.shape)
