import torch
import torch.nn as nn
from typing import List, Union
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from utils import get_device
from torchviz import make_dot


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


class IdentityConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(channels, channels,
                         kernel_size=3, padding="same", bias=False)
        identity_matrix = torch.eye(channels).view(channels, channels, 1, 1)
        # print("Identity matrix type: ", identity_matrix.type())
        with torch.no_grad():
            conv.weight.copy_(identity_matrix)
        self.conv = nn.Sequential(conv,
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU())

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)

class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int) -> None:
        super().__init__()
        convs = list()
        self.count = 0
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        convs.extend(
            [nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding="same"),
             nn.BatchNorm2d(num_features=out_channels),
             nn.ReLU()]
        )
        self.convs = nn.ModuleList(convs)
        self.device = get_device()

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x

    def add_layer(self):
        if self.count < 3:
            new_layer = IdentityConvLayer(
                channels=self.out_channels).to(self.device)
            # self.convs.insert(len(self.convs)-1, new_layer)
            self.convs.append(new_layer)
            self.count += 1


class DynamicCNN(nn.Module):
    def __init__(self, channels_list: List[int], n_classes: int, dropout: float = 0.) -> None:
        super().__init__()
        if not channels_list:
            raise ValueError("Channels list should not be empty")
        blocks = []
        self.n_classes = n_classes
        self.dropout = dropout
        self.device = get_device()

        for i in range(len(channels_list)-1):
            blocks.extend([ConvBlock(in_channels=channels_list[i],
                                     kernel_size=3,
                                     out_channels=channels_list[i+1]),
                          nn.MaxPool2d(3)])

        self.convs = nn.ModuleList(blocks)
        self.fc = nn.Sequential(
            MLP(90, 20, dropout=dropout),
            nn.BatchNorm1d(20),
            MLP(20, out_features=n_classes, dropout=dropout, is_output_layer=True)
        )
        self.one_by_one_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_list[-1], out_channels=10, kernel_size=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x)

        x = self.one_by_one_conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def compute_fisher_information(self,
                                   dataloader: DataLoader,
                                   criterion: nn.CrossEntropyLoss):
        fisher_information = {}

        for name, param in self.named_parameters():
            fisher_information[name] = torch.zeros_like(param)

        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            self.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad.pow(
                        2) / len(dataloader)
        self.train()
        return fisher_information

    def compute_natural_expansion_score(self, dataloader: DataLoader, criterion: nn.CrossEntropyLoss):
        fisher_information = self.compute_fisher_information(dataloader,
                                                             criterion)
        natural_expansion_score = 0.0
        # Setting to eval mode
        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            self.zero_grad()
            loss.backward()

            # Accumulate the natural expansion score
            for name, param in self.named_parameters():
                if param.grad is not None and name in fisher_information:
                    # Fisher inverse component for this parameter
                    fisher_inv = 1.0 / (fisher_information[name] + 1e-5)

                    # Updating the natural expansion scrore
                    # score = g^T * F^(-1) * g = g**2 *F^(-1)
                    natural_expansion_score += torch.sum(
                        param.grad ** 2 * fisher_inv)

        num_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        # print(f"num params: {num_params}")
        # print(f"np before div{natural_expansion_score}")
        natural_expansion_score /= (len(dataloader) * num_params)
        # print(f"Natural expansion score: {natural_expansion_score}")

        # Setting back to train mode
        self.train()
        return natural_expansion_score.item()

    def expand_if_necessary(self, dataloader: DataLoader,
                            threshold: float,
                            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()):
        optimal_index = self.find_optimal_location(
            dataloader=dataloader, threshold=threshold, criterion=criterion)
        if optimal_index is not None:
            print("adding layer")
            self.expand(optimal_index)
            print("added layer")

    def expand(self, optimal_index: int) -> None:
        self.convs[optimal_index].add_layer()

    def needs_expansion(self, dataloader: DataLoader, threshold: float, criterion: nn.CrossEntropyLoss) -> bool:
        score = self.compute_natural_expansion_score(dataloader=dataloader,
                                                     criterion=criterion)
        return score > threshold

    def find_optimal_location(self, dataloader: DataLoader, threshold: float, criterion: nn.CrossEntropyLoss) -> int:
        scores = []
        current_score = self.compute_natural_expansion_score(
            dataloader=dataloader, criterion=criterion)

        conv_block_indices = []
        for index, module in enumerate(self.convs):
            if isinstance(module, ConvBlock):
                conv_block_indices.append(index)

        for index in conv_block_indices:  # range(num_convs-1):
            temp_model = copy.deepcopy(self)
            # print(f"CONVS: {temp_model.convs[index]}")
            temp_model.convs[index].add_layer()
            new_score = temp_model.compute_natural_expansion_score(
                dataloader, criterion)
            # print(f"score at index {index}: {new_score}")
            if (new_score/current_score) > threshold:
                scores.append({
                    index: new_score
                })
            del temp_model
        # print(f"Scores: {scores}")
        if scores:
            max_dict = max(scores, key=lambda d: sum(d.values()))
            optimal_index = list(max_dict)[0]
        else:
            optimal_index = None
        # print(f"optimal_index: {optimal_index}")
        return optimal_index