import torch
import torch.nn as nn
from patchEmbedding import PatchEmbedding
from transformerBlock import TransformerBlock


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 num_classes: int,
                 in_channels: int = 3):
        """
        Initializes the Vision Transformer model
        Args:
            image_size: size of the input image
            patch_size: size of a single patch
            embed_dim: dimension of the embedding vectors
            num_heads: number of heads in the transformer block
            num_layers: number of transformer blocks
            num_classes: number of output classes for classification
            in_channels: number of channels in the input image, defaults to 3 for RGB
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, embed_dim, in_channels)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.Linear(embed_dim//2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass
        Args:
            x: The input tensor of the image

        Returns: The classification logits
        """
        x = self.patch_embedding(x)
        x = self.transformer_blocks(x)
        cls_token_embedding = x[:, 0]
        out = self.classifier(cls_token_embedding)
        return out


if __name__ == "__main__":
    classifier = VisionTransformer(
        image_size=64, patch_size=16, embed_dim=8, num_heads=8, num_layers=4, num_classes=3)
    rand = torch.ones(32, 3, 64, 64)
    print(classifier(rand).shape)
