import torch.nn as nn
import torch


class MLP(nn.Module):
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
                      nn.ReLU(),
                      nn.Dropout(dropout)])

        self.fc = nn.ModuleList(fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc:
            x = layer(x)
        return x


if __name__ == "__main__":
    random_tensor = torch.rand(2, 10)
    model = MLP(in_features=10, out_features=5, is_output_layer=False)
    print(model)
    pred = model(random_tensor)
    print(pred.shape)
