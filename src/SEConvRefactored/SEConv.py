import torch
import torch.nn as nn
from typing import List
from utils import get_device


class IdentityConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=1, padding=0, bias=False)
        identity_matrix = torch.eye(channels).view(channels, channels, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(identity_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SelfExpandingCNN(nn.Module):
    def __init__(self,
                 channels_list: List[int],
                 n_classes: int,
                 input_shape=(1, 28, 28)) -> None:
        super().__init__()
        if not channels_list:
            raise ValueError("channels_list must not be empty")

        self.layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.layers.append(
                nn.Conv2d(channels_list[i], channels_list[i+1], kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))

        self.input_shape = input_shape
        self._update_fc(input_shape, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def add_identity_layer(self, insert_index: int) -> None:
        if insert_index <= 0 or insert_index >= len(self.layers):
            raise IndexError("insert_index is out of bounds")

        # Get device and data type of existing layers
        existing_device = next(self.parameters()).device
        existing_dtype = next(self.parameters()).dtype

        channels = self._get_channels_before_index(insert_index)
        identity_layer = IdentityConvLayer(channels)

        # Set the device and data type of the new layer to match the existing network
        identity_layer = identity_layer.to(
            device=existing_device, dtype=existing_dtype)

        self.layers.insert(insert_index, identity_layer)
        self.layers.insert(insert_index + 1, nn.ReLU())
        self._update_fc(self.input_shape, self.fc.out_features)

    def _update_fc(self, input_shape, n_classes):
        fc_input_size = self._get_conv_output(input_shape)
        self.fc = nn.Linear(fc_input_size, n_classes)

    def _get_conv_output(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.autograd.Variable(torch.rand(1, *input_shape))
            output = dummy_input
            for layer in self.layers:
                output = layer(output)
            return int(torch.flatten(output, 1).size(1))

    def compute_fisher_information(self, dataloader, criterion) -> dict:
        fisher_information = {}
        for name, param in self.named_parameters():
            fisher_information[name] = torch.zeros_like(param)

        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(get_device()), labels.to(get_device())
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

    def compute_natural_expansion_score(self, dataloader, criterion) -> float:
        fisher_information = self.compute_fisher_information(
            dataloader, criterion)
        natural_grad_approx = 0
        for name, param in self.named_parameters():
            if param.grad is not None and name in fisher_information:
                fisher_diag = fisher_information[name]
                natural_grad_approx += (param.grad.pow(2) /
                                        (fisher_diag + 1e-5)).sum()
        return natural_grad_approx.item()

    def _compute_score_with_temp_layer(self, dataloader, criterion, insert_index: int):
        identity_layer = IdentityConvLayer(
            self.layers[insert_index - 1].out_channels).to(get_device())
        self.layers.insert(insert_index, identity_layer)
        score = self.compute_natural_expansion_score(dataloader, criterion)
        del self.layers[insert_index]
        return score

    def find_optimal_expansion_location(self, dataloader, criterion):
        max_relative_increase = 0
        optimal_index = None
        current_score = self.compute_natural_expansion_score(
            dataloader, criterion)

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], 'out_channels'):
                score_with_layer = self._compute_score_with_temp_layer(
                    dataloader, criterion, i + 1)
                # Added 1e-5 to avoid division by zero
                relative_increase = score_with_layer / \
                    (current_score + 1e-5) - 1

                if relative_increase > max_relative_increase:
                    max_relative_increase = relative_increase
                    optimal_index = i + 1

        return optimal_index

    def expand_if_necessary(self, dataloader, criterion, threshold: float):
        optimal_index = self.find_optimal_expansion_location(
            dataloader, criterion)
        if optimal_index is not None:
            current_score = self.compute_natural_expansion_score(
                dataloader, criterion)
            score_with_layer = self._compute_score_with_temp_layer(
                dataloader, criterion, optimal_index)
            relative_increase = score_with_layer / (current_score + 1e-5) - 1

            if relative_increase > threshold:
                self.add_identity_layer(optimal_index)


# Utility function to get the device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")