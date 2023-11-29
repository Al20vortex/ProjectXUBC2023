import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import List
import warnings
warnings.filterwarnings("ignore")


class SelfAttention(nn.Module):
    """
    An implementation of self attention in pytorch 
    """

    def __init__(self, dim: int) -> None:
        """
        Takes in a tensor and spits out a tensor of the same dimension after undergoing self attention
        NOTE: dim is the dimension of the embedding.
        Args:
            dim (torch.Tensor): The input dimension of the embedding
        """
        super(SelfAttention, self).__init__()
        self.dim = dim

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        scores = torch.matmul(Q, K.T)
        scores_normalized = scores / np.sqrt(self.dim)
        softed = nn.Softmax(dim=-1)(scores_normalized)
        z = torch.matmul(softed, V)
        return z


class MultiHeadedAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """
        Performs a multiheaded attention operation on input data

        Args:
            dim (int): The dimension of the input vectors
            num_heads (int, optional): The number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * self.num_heads == dim, "The dimensions must be divisible by num_heads"

        self.query_layers = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Performs a forward pass throught the network

        Args:
            X (torch.tensor): The input tensor before a multiheaded attention operation

        Returns:
            torch.tensor: The output tensor after a multiheaded attention operation
        """
        attention_heads = []
        for i in range(self.num_heads):
            queries = self.query_layers[i](X)
            keys = self.key_layers[i](X)
            values = self.value_layers[i](X)

            attention = SelfAttention(self.head_dim)(queries, keys, values)
            attention_heads.append(attention)

        print(attention_heads)
        attention_heads = torch.cat(attention_heads, dim=-1)
        output = self.fc(attention_heads)
        return output


if __name__ == "__main__":
    att = MultiHeadedAttention(dim=16, num_heads=8)
    tensor = torch.ones(10, 16)
    out = att(tensor)
    print(out.shape)
