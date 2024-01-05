import torch
import torch.nn as nn
from typing import List, Union
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import copy
from utils import check_network_consistency, get_device
from torchviz import make_dot

device = get_device()

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
        with torch.no_grad():
            conv.weight.copy_(identity_matrix)
        self.conv = nn.Sequential(conv,
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU()).to(device)

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
             nn.ReLU(),
             nn.MaxPool2d(2)
            ]
        )
        self.convs = nn.ModuleList(convs)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x)
        return x
    
    def add_layer(self):
        if self.count < 3:
            new_layer = IdentityConvLayer(
                channels=self.out_channels).to(self.device)
            self.convs.insert(len(self.convs)-2, new_layer)
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
            ])
                        #   nn.MaxPool2d(2)]) # TODO keep maxpooling here?

        self.convs = nn.ModuleList(blocks)
        self.fc = nn.Sequential(
            MLP(640, 20, dropout=dropout), #TODO dynamically calculate MLP dims
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

    def compute_fisher_information(self, dataloader: DataLoader, criterion: nn.CrossEntropyLoss):
        fisher_information = {}
        for name, param in self.named_parameters():
            fisher_information[name] = torch.zeros_like(param)

        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            self.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad.pow(2) / len(dataloader)
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

    def upgrade_block(self, index, upgrade_factor):
        block = self.convs[index]
        new_out_channels = int(block.out_channels * upgrade_factor)

        for layer in block.convs:
            if isinstance(layer, nn.Conv2d):
                self.upgrade_conv_layer(layer, new_out_channels)
            elif isinstance(layer, nn.BatchNorm2d):
                self.upgrade_batchnorm_layer(layer, new_out_channels)
            elif isinstance(layer, IdentityConvLayer):
                self.upgrade_identity_conv_layer(layer, new_out_channels)

        block.out_channels = new_out_channels

        # If the next Block is a ConvBlock
        if index + 1 < len(self.convs) and isinstance(self.convs[index + 1], ConvBlock):
            next_block = self.convs[index + 1]
            self.upgrade_next_block_input(next_block, new_out_channels)
        elif index == len(self.convs) - 1:  # Upgrading the last ConvBlock
            self.upgrade_one_by_one_conv(new_out_channels)

    def upgrade_one_by_one_conv(self, new_in_channels):
        old_weights = self.one_by_one_conv[0].weight.data
        old_out_channels, old_in_channels, kernel_height, kernel_width = old_weights.shape

        new_weights = torch.zeros((old_out_channels, new_in_channels, kernel_height, kernel_width), device=self.device)
        new_weights[:, :old_in_channels, :, :] = old_weights

        self.one_by_one_conv[0].in_channels = new_in_channels
        self.one_by_one_conv[0].weight = nn.Parameter(new_weights)

        if self.one_by_one_conv[0].bias is not None:
            self.one_by_one_conv[0].bias = nn.Parameter(self.one_by_one_conv[0].bias.data)

    def upgrade_conv_layer(self, layer, new_out_channels):
        old_weights = layer.weight.data
        old_out_channels, old_in_channels, kernel_height, kernel_width = old_weights.shape

        new_weights = torch.randn((new_out_channels, old_in_channels, kernel_height, kernel_width), device=self.device) * 0.01
        new_weights[:old_out_channels, :, :, :] = old_weights

        new_bias = None
        if layer.bias is not None:
            old_bias = layer.bias.data
            new_bias = torch.zeros(new_out_channels, device=self.device)
            new_bias[:old_out_channels] = old_bias

        layer.out_channels = new_out_channels
        layer.weight = nn.Parameter(new_weights)
        layer.bias = nn.Parameter(new_bias) if new_bias is not None else None
   
    def upgrade_identity_conv_layer(self, identity_layer, new_channels):
        conv_layer = identity_layer.conv[0]  # Accessing the Conv2d layer in IdentityConvLayer
        device = conv_layer.weight.device  # Ensure we're using the same device as the layer's weights

        old_weights = conv_layer.weight.data
        old_out_channels, old_in_channels, kernel_height, kernel_width = old_weights.shape

        # Adjust weights for new channel dimensions
        new_weights = torch.zeros((new_channels, new_channels, kernel_height, kernel_width), device=device)
        min_channels = min(old_in_channels, new_channels)
        new_weights[:min_channels, :min_channels, :, :] = old_weights[:min_channels, :min_channels, :, :]

        # Update Conv2d layer properties
        conv_layer.out_channels = new_channels
        conv_layer.in_channels = new_channels
        conv_layer.weight = nn.Parameter(new_weights)

        # Adjust BatchNorm layer following the Conv2d layer
        identity_layer.conv[1] = nn.BatchNorm2d(new_channels).to(device)

    def upgrade_next_block_input(self, block, new_in_channels):
        first_layer = block.convs[0]
        if isinstance(first_layer, nn.Conv2d):
            old_weights = first_layer.weight.data
            old_out_channels, old_in_channels, kernel_height, kernel_width = old_weights.shape

            # New weights: existing weights are preserved, new weights are randomly initialized
            new_weights = torch.randn((old_out_channels, new_in_channels, kernel_height, kernel_width), device=self.device) * 0.01
            new_weights[:, :old_in_channels, :, :] = old_weights

            # Update Conv2d layer's in_channels attribute and weights tensor
            first_layer.in_channels = new_in_channels
            first_layer.weight = nn.Parameter(new_weights)

            # Maintain the bias tensor if it exists
            if first_layer.bias is not None:
                first_layer.bias = nn.Parameter(first_layer.bias.data)

    def upgrade_batchnorm_layer(self, layer, new_out_channels):
        old_out_channels = layer.num_features

        new_running_mean = torch.zeros(new_out_channels, device=self.device)
        new_running_var = torch.ones(new_out_channels, device=self.device)

        if old_out_channels < new_out_channels:
            new_running_mean[:old_out_channels] = layer.running_mean
            new_running_var[:old_out_channels] = layer.running_var
        else:
            new_running_mean = layer.running_mean
            new_running_var = layer.running_var

        layer.num_features = new_out_channels
        layer.running_mean = new_running_mean
        layer.running_var = new_running_var
        layer.weight = nn.Parameter(torch.ones(new_out_channels, device=self.device))
        layer.bias = nn.Parameter(torch.zeros(new_out_channels, device=self.device)) # Get the block to upgrade


    def find_optimal_action(self, dataloader: DataLoader, threshold: float, upgrade_factor: float, criterion: nn.CrossEntropyLoss) -> Union[str, int]:
        best_score = 0
        best_action = None
        best_index = None
        subset_indices = range(128)  # Adjust the range as needed
        subset = Subset(dataloader.dataset, subset_indices)

        # Compute natural expansion score using only a subset of the full train set
        subset_dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)
        current_score = self.compute_natural_expansion_score(dataloader=subset_dataloader, criterion=criterion)

        # Evaluate adding new layers
        for index, module in enumerate(self.convs):
            if isinstance(module, ConvBlock):
                temp_model = copy.deepcopy(self)
                temp_model.convs[index].add_layer()
                new_score = temp_model.compute_natural_expansion_score(dataloader, criterion)
                if (new_score/current_score) > threshold and new_score > best_score:
                    best_score = new_score
                    best_action = "add_layer"
                    best_index = index
                del temp_model

        # Evaluate upgrading existing blocks
        for index, module in enumerate(self.convs):
            if isinstance(module, ConvBlock):
                temp_model = copy.deepcopy(self)
                temp_model.upgrade_block(index, upgrade_factor)
                new_score = temp_model.compute_natural_expansion_score(dataloader, criterion)
                if (new_score/current_score) > threshold and new_score > best_score:
                    best_score = new_score
                    best_action = "upgrade_block"
                    best_index = index
                del temp_model

        return best_action, best_index
    
    def expand_if_necessary(self, 
                            dataloader: DataLoader,
                            threshold: float,
                            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
                            upgrade_factor = 1.5):
        optimal_action, optimal_index = self.find_optimal_action(
            dataloader=dataloader, threshold=threshold, upgrade_factor=upgrade_factor, criterion=criterion)
        if optimal_action == "add_layer":
            print("Adding layer at index", optimal_index)
            self.convs[optimal_index].add_layer()
            check_network_consistency(self)
        elif optimal_action == "upgrade_block":
            print("Upgrading block at index", optimal_index)
            self.upgrade_block(optimal_index, upgrade_factor)
            check_network_consistency(self)
        else:
            print("No expansion or upgrade necessary at this time")