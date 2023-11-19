import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, patch_size: int, embed_dim: int, in_channels: int = 3):
        """
        Embeds an image using patches
        Args:
            patch_size: The size of a single patch
            embed_dim: The dimension of the embedding vectors
            in_channels: The number of channels, defaults to 3 assuming RGB
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = nn.Flatten(2)(x)
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":
    rand = torch.ones(20, 3, 64, 64)  # .unsqueeze(0)
    emb = PatchEmbedding(patch_size=16, in_channels=3, embed_dim=8)
    print(emb(rand).shape)
