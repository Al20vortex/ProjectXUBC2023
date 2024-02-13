import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A fully connected layer followed by a batchnorm, activation and a dropout layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float = 0.0,
                 is_output_layer: bool = False):
        super().__init__()
        fc = []
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        fc.append(layer)
        if not is_output_layer:
            fc.extend([nn.BatchNorm1d(num_features=out_features),
                      nn.LeakyReLU(0.2),
                      nn.Dropout(dropout)])

        self.fc = nn.ModuleList(fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc:
            x = layer(x)
        return x
