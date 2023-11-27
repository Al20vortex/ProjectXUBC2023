import torch
import torch.nn as nn


class NN(nn.Module):

    def __init__(self, in_features: int, n_classes: int):
        """
        Neural Network, wants to check mutual information between layers

        Args:
            in_features (int): Number of features
            n_classes (int): Number of classes
        """
        super().__init__()
        self.layer_1 = nn.Linear(in_features=in_features, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=30)
        self.layer_3 = nn.Linear(in_features=30, out_features=20)
        self.layer_4 = nn.Linear(in_features=20, out_features=10)
        self.layer_5 = nn.Linear(in_features=10, out_features=n_classes)

        self.layers = nn.Sequential(
            self.layer_1, nn.ReLU(), self.layer_2, nn.ReLU(
            ), self.layer_3, nn.ReLU(), self.layer_4, nn.ReLU(),
            self.layer_5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input 

        Returns:
            torch.Tensor: output
        """
        x = self.layers(x)
        return x


if __name__ == "__main__":
    model = NN(in_features=10, n_classes=5)
    print(model)
    model.layer_5 = nn.Linear(10, 20)
    print(model)
    a = torch.rand(20, 10)
    out = model(a)
    print(model)
    print(out.shape)
