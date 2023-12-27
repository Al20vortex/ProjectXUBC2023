import torch
import torch.nn as nn
from typing import List, Union
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from .utils import get_device
from torchviz import make_dot
from .perceptron import *
from .conv_block import *


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

        # Normalizing by the number of data points(TODO: consider number of parameters?)
        natural_expansion_score /= len(dataloader.dataset)
        # Setting back to train mode
        self.train()
        return natural_expansion_score.item()

    def expand_if_necessary(self, dataloader: DataLoader,
                            threshold: float,
                            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()):
        # breakpoint()

        if self.needs_expansion(dataloader, threshold, criterion):
            optimal_index = self.find_optimal_location(
                dataloader=dataloader, threshold=threshold, criterion=criterion)
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

        num_convs = len(self.convs)

        conv_block_indices = []
        for index, module in enumerate(self.convs):
            if isinstance(module, ConvBlock):
                conv_block_indices.append(index)

        for index in conv_block_indices:  # range(num_convs-1):
            temp_model = copy.deepcopy(self)
            print(f"CONVS: {temp_model.convs[index]}")
            temp_model.convs[index].add_layer()
            new_score = temp_model.compute_natural_expansion_score(
                dataloader, criterion)
            scores.append(new_score)
            del temp_model
        print(f"Scores: {scores}")
        optimal_index = torch.argmax(torch.Tensor(scores))
        print(
            f"Optimal Index: {optimal_index}, type: {type(optimal_index.item())}")
        return optimal_index.item()


if __name__ == "__main__":
    model = DynamicCNN([3, 4, 4], 10, 0.2)
    random_tensor = torch.rand(32, 3, 28, 28)
    yhat = model(random_tensor)
    # print(model(random_tensor).shape)
    # make_dot(yhat, params=dict(list(model.named_parameters()))
    #  ).render("Computation_graph", format="png")
    print(yhat)
    # print(a.convs)
    # a.expand(1)
    # print(a.convs)
    # print(len(a.convs))
    # # print(type(a.convs))
    # for i in a.convs:
    #     if isinstance(i, ConvBlock):
    #         print(i)
