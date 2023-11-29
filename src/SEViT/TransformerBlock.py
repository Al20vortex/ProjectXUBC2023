import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Initializes the transformer block, adds layer norm and skip connection
        Args:
            embed_dim: The dimension of the embedding vectors
            num_heads: The number of heads in the transformer block
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.fcn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Dropout1d()
            nn.Dropout()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass
        Args:
            X: The input tensor

        Returns: The transformer block after attention, layernorm and fcn
        """
        attention_output, _ = self.attention(query=X, key=X, value=X)
        x1 = self.layer_norm_1(attention_output + X)
        x = self.fcn(x1)
        x = self.layer_norm_2(x + x1)
        return x


if __name__ == "__main__":
    rand = torch.ones((2, 10, 8))
    a = TransformerBlock(embed_dim=8, num_heads=8)
    print(a)
    print(a(rand).shape)
