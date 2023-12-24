import torch
import torch.nn as nn
from positionalEncoding import PositionalEncoding
print(positionalEncoding)


class PatchEmbedding(nn.Module):

    def __init__(self, image_size: int, patch_size: int, embed_dim: int, in_channels: int = 3):
        """
        Embeds an image using patches
        Args:
            patch_size: The size of a single patch
            embed_dim: The dimension of the embedding vectors
            in_channels: The number of channels, defaults to 3 assuming RGB
        """
        super().__init__()
        assert (image_size %
                patch_size) == 0, "Image size not divisible by patch size"
        num_patches = (image_size // patch_size) ** 2

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        # self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.positional_embeddings = PositionalEncoding(
            embed_dim, num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performa the forward pass
        Args:
            x: The input tensor of the image

        Returns: The input tensor ready for attention

        """
        x = self.conv(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embeddings.generate().unsqueeze(0)
        return x


if __name__ == "__main__":
    rand = torch.ones(20, 3, 64, 64)  # .unsqueeze(0)
    emb = PatchEmbedding(image_size=64, patch_size=16,
                         in_channels=3, embed_dim=8)
    print(emb(rand).shape)
