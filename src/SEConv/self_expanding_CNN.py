import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class IdentityConvLayer(nn.Module):

    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=1, padding=0, bias=False)
        self.conv.weight.data = torch.eye(
            channels).view(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SelfExpandingCNN(nn.Module):
    def __init__(self, channels_list: List[int], n_classes: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.layers.append(
                nn.Conv2d(channels_list[i], channels_list[i+1], kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))

        # Assuming 4x4 feature map size before FC layer
        fc_input_size = self._get_conv_output()
        self.fc = nn.Linear(fc_input_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def add_identity_layer(self) -> None:
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Conv2d):
                channels = layer.out_channels
                break
        insert_index = len(self.layers) - 3
        identity_layer = IdentityConvLayer(channels)
        self.layers.insert(insert_index, identity_layer)
        self.layers.insert(insert_index+1, nn.ReLU())

    def _get_conv_output(self, shape=(1, 28, 28)):
        """
        Pass a dummy input through the convolutional layers to get the output size.

        Args:
            shape (tuple): The shape of the input (C, H, W).

        Returns:
            int: The size of the output after it has been flattened.
        """
        dummy_input = torch.autograd.Variable(torch.rand(1, *shape))
        output = dummy_input
        for layer in self.layers:
            output = layer(output)
        return int(torch.flatten(output, 1).size(1))

    def compute_fisher_information(self, dataloader: torch.Tensor, criterion: torch.Tensor) -> dict:
        """
        Computes the fisher matrix

        Args:
            dataloader (torch.Tensor): The dataloader
            criterion (torch.Tensor): The criterion. 

        Returns:
            dict: The fisher information
        """
        fisher_information = {}
        for name, param in self.named_parameters():
            fisher_information[name] = torch.zeros_like(param)

        self.eval()
        for inputs, labels in dataloader:
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            self.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += (param.grad **
                                                 2) / len(dataloader)
        return fisher_information

    def compute_natural_expansion_score(self, dataloader: torch.Tensor, criterion: torch.Tensor) -> float:
        """
        Computes the natural expansion score as per the paper SelfExpandinNeuralNetworks

        Args:
            dataloader (torch.Tensor): The dataloader baby
            criterion (torch.Tensor): The criterion baby

        Returns:
            float: The score baby
        """
        fisher_information = self.compute_fisher_information(
            dataloader, criterion)

        natural_grad_approx = 0
        for name, param in self.named_parameters():
            if param.grad is not None and name in fisher_information:
                fisher_diag = fisher_information[name]
                natural_grad_approx += (param.grad **
                                        2 / (fisher_diag + 1e-5)).sum()

        return natural_grad_approx.item()

    def expand_if_necessary(self, dataloader: torch.Tensor, criterion: torch.Tensor, threshold: float) -> None:
        """
        Expands the network if the expansion score is above the threshold

        Args:
            dataloader (torch.Tensor): The dataloader baby
            criterion (torch.Tensor): the criterion baby
            threshold (float): Threshold baby
        """
        score = self.compute_natural_expansion_score(dataloader, criterion)
        print(f"Score: {score}")
        if score > threshold:
            self.add_identity_layer()
