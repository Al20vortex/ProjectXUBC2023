import torch
import torch.nn as nn
from typing import List, Union
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import copy
from utils import check_network_consistency, count_parameters, get_device
import math

device = get_device()
NES_REG = 5e-5

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
                      nn.LeakyReLU(0.2),
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
        
        # Creating an identity matrix with added noise
        identity_matrix = torch.eye(channels).view(channels, channels, 1, 1)
        noise = torch.randn(identity_matrix.shape) * 0.01
        identity_matrix_with_noise = identity_matrix + noise
        with torch.no_grad():
            conv.weight.copy_(identity_matrix)
        self.conv = nn.Sequential(conv,
                                  nn.BatchNorm2d(channels),
                                  nn.LeakyReLU(0.2)).to(device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)

class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            pooling_amount: int,
            dropout: float) -> None:
        super().__init__()
        convs = list()
        self.count = 0
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        convs.extend(
            [nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding="same"),
             nn.Dropout(self.dropout),
             nn.BatchNorm2d(num_features=out_channels),
             nn.LeakyReLU(0.2),
             nn.MaxPool2d(pooling_amount)
            ]
        )
        self.convs = nn.ModuleList(convs)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x)
        return x
    
    def add_layer(self):
        if self.count < 6:
            new_layer = IdentityConvLayer(
                channels=self.out_channels).to(self.device)
            self.convs.insert(len(self.convs)-2, new_layer)
            self.count += 1

class DynamicCNN(nn.Module):
    def __init__(self, channels_list: List[int], n_classes: int, dropout: float = 0., image_size: int = 32, pooling_stride=2) -> None:
        super().__init__()
        if not channels_list:
            raise ValueError("Channels list should not be empty")
        blocks = []
        self.pooling_stride = pooling_stride
        self.n_classes = n_classes
        self.dropout = dropout
        self.device = device
        self.dropout = dropout

        for i in range(len(channels_list)-1):
            blocks.extend([ConvBlock(in_channels=channels_list[i],
                                     kernel_size=3,
                                     out_channels=channels_list[i+1],
                                     pooling_amount=pooling_stride,
                                     dropout=self.dropout),
            ])

        self.convs = nn.ModuleList(blocks)

        self.one_by_one_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_list[1] + channels_list[-1], out_channels=n_classes, kernel_size=1),
            nn.LeakyReLU(0.2)
        )

        for i in range(len(channels_list) - 1):
            # Assuming each ConvBlock includes a max pooling layer with kernel_size=3
            image_size = math.floor((image_size - 1) / pooling_stride)
            if pooling_stride == 2:
                image_size+=1

        mlp_input_features = image_size * image_size * n_classes

        self.fc = nn.Sequential(
            MLP(mlp_input_features, 20, dropout=dropout),
            nn.BatchNorm1d(20),
            MLP(20, out_features=n_classes, dropout=dropout, is_output_layer=True)
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs[0](x)
        x_skip = x
        for layer in self.convs[1:]:
            x = layer(x)
            x_skip = nn.MaxPool2d(self.pooling_stride)(x_skip)
        final_x = torch.cat((x, x_skip), dim=1)
        x = self.one_by_one_conv(final_x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def compute_fisher_information(self, dataloader: DataLoader, criterion: nn.CrossEntropyLoss):
        fisher_information = {name: torch.zeros_like(param) for name, param in self.named_parameters()}

        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            self.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad.pow(2)

        self.train()
        return fisher_information

    def compute_natural_expansion_score(self, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, current_param_count: int):
        fisher_information = self.compute_fisher_information(dataloader, criterion)
        natural_expansion_score = 0.0
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            self.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None and name in fisher_information:
                    fisher_inv = 1.0 / (fisher_information[name] + 1e-5)
                    natural_expansion_score += torch.sum(param.grad ** 2 * fisher_inv)
        natural_expansion_score /= num_params

        param_increase = num_params - current_param_count

        # natural_expansion_score = natural_expansion_score - L1_REG*math.log(param_increase)
        natural_expansion_score = natural_expansion_score*math.pow(math.e, -NES_REG*param_increase)
        self.train()
        return natural_expansion_score.item()

    def upgrade_block(self, index, upgrade_amount):
        block = self.convs[index]
        new_out_channels = int(block.out_channels + upgrade_amount)

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
            self.upgrade_one_by_one_conv()
        
        # if upgrading block zero, upgrade also the one by one because of the skip connection
        if index == 0:
            self.upgrade_one_by_one_conv()

    def upgrade_one_by_one_conv(self):
        # Sum of the output channels of the first and last ConvBlock
        total_in_channels = self.convs[0].out_channels + self.convs[-1].out_channels

        # Accessing the one_by_one_conv layer
        one_by_one_conv = self.one_by_one_conv[0]

        # Adjusting the input channels of the one_by_one_conv layer
        old_weights = one_by_one_conv.weight.data
        old_out_channels, _, _, _ = old_weights.shape
        new_weights = torch.zeros((old_out_channels, total_in_channels, 1, 1), device=self.device)

        # Ensuring that the existing weights are properly allocated in the new weight matrix
        new_weights[:, :self.convs[0].out_channels, :, :] = old_weights[:, :self.convs[0].out_channels, :, :]
        new_weights[:, self.convs[0].out_channels:, :, :] = old_weights[:, -self.convs[-1].out_channels:, :, :]

        # Updating the one_by_one_conv layer parameters
        one_by_one_conv.in_channels = total_in_channels
        one_by_one_conv.weight = nn.Parameter(new_weights)
        if one_by_one_conv.bias is not None:
            one_by_one_conv.bias = nn.Parameter(one_by_one_conv.bias.data)

    def upgrade_conv_layer(self, layer, new_out_channels):
        """
        Upgrades a convolutional layer, ensuring correct parameter registration.
        """
        old_weights = layer.weight.data
        old_out_channels, old_in_channels, kernel_height, kernel_width = old_weights.shape

        # Initializing new weights
        new_weights = torch.randn((new_out_channels, old_in_channels, kernel_height, kernel_width), device=self.device) * 0.01
        new_weights[:old_out_channels, :, :, :] = old_weights

        # Handling bias
        new_bias = None
        if layer.bias is not None:
            old_bias = layer.bias.data
            new_bias = torch.zeros(new_out_channels, device=self.device)
            new_bias[:old_out_channels] = old_bias

        # Re-registering parameters
        layer.out_channels = new_out_channels
        layer.weight = nn.Parameter(new_weights)
        layer.bias = nn.Parameter(new_bias) if new_bias is not None else None
        
    def upgrade_identity_conv_layer(self, identity_layer, new_channels):
        """
        Upgrades an identity convolution layer, ensuring proper parameter handling.
        """
        conv_layer = identity_layer.conv[0]
        old_weights = conv_layer.weight.data
        old_out_channels, old_in_channels, kernel_height, kernel_width = old_weights.shape

        # Initializing new weights
        new_weights = torch.zeros((new_channels, new_channels, kernel_height, kernel_width), device=self.device)
        min_channels = min(old_out_channels, new_channels)
        new_weights[:min_channels, :min_channels, :, :] = old_weights

        # Re-registering parameters
        conv_layer.out_channels = new_channels
        conv_layer.in_channels = new_channels
        conv_layer.weight = nn.Parameter(new_weights)

        # Adjusting BatchNorm layer
        identity_layer.conv[1] = nn.BatchNorm2d(new_channels).to(self.device)

    def upgrade_next_block_input(self, block, new_in_channels):
        """
        Upgrades the input channels of the next block in the network.
        """
        first_layer = block.convs[0]
        old_weights = first_layer.weight.data
        old_out_channels, _, kernel_height, kernel_width = old_weights.shape

        # Initializing new weights
        new_weights = torch.randn((old_out_channels, new_in_channels, kernel_height, kernel_width), device=self.device) * 0.01
        new_weights[:, :first_layer.in_channels, :, :] = old_weights

        # Re-registering parameters
        first_layer.in_channels = new_in_channels
        first_layer.weight = nn.Parameter(new_weights)

        # Handling bias
        if first_layer.bias is not None:
            first_layer.bias = nn.Parameter(first_layer.bias.data)
            
    def upgrade_batchnorm_layer(self, layer, new_out_channels):
        """
        Upgrades a batch normalization layer, ensuring correct parameter registration.
        """
        new_running_mean = torch.zeros(new_out_channels, device=self.device)
        new_running_var = torch.ones(new_out_channels, device=self.device)

        # Copying old parameters
        new_running_mean[:layer.num_features] = layer.running_mean
        new_running_var[:layer.num_features] = layer.running_var

        # Re-registering parameters
        layer.num_features = new_out_channels
        layer.running_mean = new_running_mean
        layer.running_var = new_running_var
        layer.weight = nn.Parameter(torch.ones(new_out_channels, device=self.device))
        layer.bias = nn.Parameter(torch.zeros(new_out_channels, device=self.device))

    def find_optimal_action(self, dataloader: DataLoader, threshold: float, upgrade_amount: int, criterion: nn.CrossEntropyLoss) -> Union[str, int]:
        best_score = 0
        best_action = None
        best_index = None

        # Compute natural expansion score using only a subset of the full train set
        subset_indices = range(512)  # Adjust the range as needed
        subset = Subset(dataloader.dataset, subset_indices)
        subset_dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)
        current_param_count = count_parameters(self)
        current_score = self.compute_natural_expansion_score(dataloader=subset_dataloader, criterion=criterion, current_param_count=current_param_count)

        # Evaluate adding new layers
        for index, module in enumerate(self.convs):
            if isinstance(module, ConvBlock):
                temp_model = copy.deepcopy(self)
                temp_model.convs[index].add_layer()
                new_score = temp_model.compute_natural_expansion_score(subset_dataloader, criterion, current_param_count)
                if (new_score/current_score) > threshold and new_score > best_score:
                    best_score = new_score
                    best_action = "add_layer"
                    best_index = index
                del temp_model

        # Evaluate upgrading existing blocks
        for index, module in enumerate(self.convs):
            if isinstance(module, ConvBlock):
                temp_model = copy.deepcopy(self)
                temp_model.upgrade_block(index, upgrade_amount)
                new_score = temp_model.compute_natural_expansion_score(subset_dataloader, criterion, current_param_count)
                print("Current score", current_score)
                print("New score", new_score)
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
                            upgrade_amount = 2) -> bool:
        optimal_action, optimal_index = self.find_optimal_action(
            dataloader=dataloader, threshold=threshold, upgrade_amount=upgrade_amount, criterion=criterion)
        if optimal_action == "add_layer":
            print("\nAdding layer at index", optimal_index)
            self.convs[optimal_index].add_layer()
            return True
        elif optimal_action == "upgrade_block":
            print("\nUpgrading block at index", optimal_index)
            self.upgrade_block(optimal_index, upgrade_amount)
            return True
        else:
            print("\nNo expansion or upgrade necessary at this time")
            return False
        