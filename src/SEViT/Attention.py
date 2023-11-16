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

        Args:
            input_dim (torch.Tensor): The input dimension of the embedding
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Performs self attention on input tensor X

        Args:
            X (tensor): the input

        Returns:
            Tensor: an output after self attention is acted on it
        """

        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)

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

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Performs a forward pass throught the network

        Args:
            X (torch.tensor): The input tensor before a multiheaded attention operation

        Returns:
            torch.tensor: The output tensor after a multiheaded attention operation
        """
        acc = list()

        for _ in range(self.num_heads):
            acc.append(SelfAttention(dim=self.dim)(X))
        attentions = torch.cat(acc, -1)
        W = nn.Linear(in_features=attentions.shape[1],
                      out_features=self.dim)
        out = W(attentions)
        return out


if __name__ == "__main__":
    random_tensor = torch.rand(9, 9)
    # print(MaskedAttention(9)(random_tensor)
    print(SelfAttention(9)(random_tensor).shape)
    # print(MultiHeadedAttention(5)(random_tensor).shape)
