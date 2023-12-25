import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from torchsummary import summary
import wandb
import random
import copy

# Define our hyperparameters
INPUT_SHAPE = (3, 64, 64)
OUTPUT_SHAPE = (1,)
POSSIBLE_SINGLE_MODULES = [nn.Conv2d]  # TODO ADD MORE
POSSIBLE_PAIR_MODULES = [nn.MaxPool2d]  # TODO ADD MORE, like CONV2d TRANSPOSE
TAU = 2.0
EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
FIXED_OUTPUT_CHANNELS = 16
L1_REG = 1e-5

# Set up hyperparameters for each possible single module
single_layer_configs = {
    "Conv2d": {
        "out_channels": [8, 16, 32, 64],
        "kernel_size": [3, 5, 7],
        "stride": [1],
    },
}

# Set up the hyperparameters for each possible paired change
paired_layers_configs = {
    # Here we need to define the possible combinations
    "Strided"
    "Conv2dStrided": "Conv2dTranspose",
    "Conv2dStrided": "Upsampling2d",
    "Conv2dStrided": "Upsampling2d",
}

wandb.init(
    # set the wandb project where this run will be logged
    project="ProjectX",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "ConvClassifier",
    "dataset": "DuckLlama",
    "tau": TAU,
    "epochs": EPOCHS,
    }
)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device =  get_device()

class NeuralNet(nn.Module):
    def __init__(self):      
        super(NeuralNet, self).__init__()
        # input layer
        self.input_layer = nn.Sequential(nn.Conv2d(3, FIXED_OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(device)
        self.layers = nn.ModuleList()

        # Initialize the output layer based on the current network configuration
        self.out = nn.Sequential(
            nn.Conv2d(FIXED_OUTPUT_CHANNELS, 4, 1, 1, 0), # here we should calculate input maps
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(4*8*8, 16),
            nn.Linear(16, OUTPUT_SHAPE[0]),
        ).to(device)

    def calculate_flattened_size(self):
        with torch.no_grad():  # No need to track gradients
            dummy_input = torch.randn(1, *INPUT_SHAPE).to(device)  # Create a dummy input on the correct device
            x = self.input_layer(dummy_input)  # Pass through the input layer

            for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    x = layer(x).to(device)  # Ensure the output is on the correct device

            # Calculate flattened size
            flattened_size = x.numel() // x.size(0)  # Total number of elements divided by batch size
            return flattened_size
        
    def calculate_output_dims(self, in_channels, x_dim, y_dim, layer):
        # Calculate the output dimensions after applying a layer
        if isinstance(layer, nn.Conv2d):
            # Calculate output dimensions for a convolutional layer
            stride = layer.stride[0]
            padding = layer.padding[0]
            dilation = layer.dilation[0]
            kernel_size = layer.kernel_size[0]

            # Formula to calculate the output size for each dimension
            x_dim = int((x_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            y_dim = int((y_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            out_channels = layer.out_channels  # Update the number of channels for the next layer
        else:
            # Handle other types of layers if needed
            out_channels = in_channels
        return out_channels, x_dim, y_dim

    # No longer in use
    def adapt_output_layer(self):
        last_conv_out_channels = self.get_last_conv_out_channels()  # Method to get the number of output channels from the last Conv2d layer
        self.out = nn.Sequential(
            nn.Conv2d(last_conv_out_channels, 3, 1, 1, 0), # here we should calculate input maps
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3*8*8, 16),
            nn.Linear(16, OUTPUT_SHAPE[0]),
        ).to(device)

    def get_last_conv_out_channels(self): # TODO unused
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Conv2d):
                return layer.out_channels
        return self.input_layer[0].out_channels  # Default to the input layer's out_channels if no conv layers are found

    def get_prev_conv_out_channels(self, position):
        # If adding at the start, use the input layer's out_channels
        if position == 0:
            return self.input_layer[0].out_channels

        # Otherwise, find the last Conv2d layer before the position
        for layer in reversed(self.layers[:position]):
            if isinstance(layer, nn.Conv2d):
                return layer.out_channels
        return self.input_layer[0].out_channels

    def insert_layer(self, position, module: nn.Module, activation_fn=nn.ReLU(), **kwargs):
        if position < 0 or position > len(self.layers):
            raise ValueError("Invalid position to insert layer")

        # Prepare for inserting the new layer
        if isinstance(module, nn.Conv2d):
            prev_out_channels = self.get_prev_conv_out_channels(position)
            kwargs['in_channels'] = prev_out_channels

        # Create and insert the new layer
        new_layer = module(**kwargs).to(device)
        identity_conv_init(new_layer)  # Initialize with identity if needed
        self.layers.insert(position, new_layer)
        self.layers.insert(position + 1, activation_fn)
        # print("Layers before adaptation", self.layers)

        # Update in_channels for subsequent Conv2d layers
        current_out_channels = new_layer.out_channels
        for i in range(position + 2, len(self.layers), 2):
            if isinstance(self.layers[i], nn.Conv2d):
                self.layers[i] = nn.Conv2d(
                    in_channels=current_out_channels, 
                    out_channels=self.layers[i].out_channels, 
                    kernel_size=self.layers[i].kernel_size, 
                    stride=self.layers[i].stride, 
                    padding=self.layers[i].padding
                ).to(device)
                current_out_channels = self.layers[i].out_channels

        # print("Layers after adaptation", self.layers)
        # Dynamically adapt the output layer based on the current network configuration
        # self.adapt_output_layer()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x.view(x.size(0))    
    
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
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
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
        num_params = self.count_parameters()
        for name, param in self.named_parameters():
            if param.grad is not None and name in fisher_information:
                fisher_diag = fisher_information[name]
                natural_grad_approx += (param.grad **
                                        2 / (fisher_diag + 1e-5)).sum()
        return natural_grad_approx.item() / num_params # TODO do we need to have this num_params


def identity_conv_init(conv_layer):
    """
    Initialize a Conv2D layer with a form of identity operation.
    Each input channel is connected to every output channel through the center of the kernel.
    This is an experimental approach and may not yield traditional identity behavior.
    """
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size[0]

    if kernel_size % 2 == 0:
        raise NotImplementedError("Identity initialization not implemented for even kernel sizes.")

    # Initialize the weights with zeros
    identity_kernel = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    center = kernel_size // 2

    for i in range(in_channels):
        for j in range(out_channels):
            identity_kernel[j, i, center, center] = 1

    # Set the weights and disable bias
    conv_layer.weight.data = identity_kernel.to(device).float()
    conv_layer.bias.data = torch.zeros(out_channels).to(device).float()

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2])),  # Resize images
    transforms.ToTensor()
    # Add more transformations if necessary
])

# Load the datasets
train_dataset = datasets.ImageFolder('llama-duck-ds/train', transform=transform)
val_dataset = datasets.ImageFolder('llama-duck-ds/val', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = NeuralNet().to(device)

# TODO params needs to be useful information for determining if the module will be beneficial
# 	Must also take into account the layer we are adding the module to.
def get_module_benefit(params, module):
    """Returns a benefit score for the module"""
    # stub
    return 0.5

def select_best_action(benefits):
    # Sort the benefits and get the best one
    benefits.sort(key=lambda x: x[0], reverse=True)
    return benefits[0]  # Returns the tuple (benefit_score, position, module_type, config)              

def update_padding_for_module(module_type, config_dict):
    """Returns the module_type and config_dicts with padding equal to:
        - same for nstrided Conv2d
        - Half the image size for strided Conv2d or Pooling
    """

    # TODO IN THE FUTURE EXPAND SUPPORT OF MORE MODULES
    if module_type.__name__ == 'Conv2d':
        # Convolution padding calculation
        if config_dict.get('stride', 1) == 1:
            kernel_size = config_dict.get('kernel_size', 0)
            config_dict['padding'] = (kernel_size - 1) // 2
        elif config_dict.get('stride', 1) == 2:
            kernel_size = config_dict.get('kernel_size', 0)
            config_dict['padding'] = (kernel_size - 2) // 2
            config_dict['padding'] = max(config_dict['padding'], 0)

    elif module_type.__name__ == 'MaxPool2d':
        # Pooling padding calculation
        kernel_size = config_dict.get('kernel_size', 0)
        if kernel_size % 2 == 0:
            config_dict['padding'] = 0  # or minimal padding as required
        else:
            config_dict['padding'] = (kernel_size - 1) // 2
    return config_dict

def calculate_expansion_score_with_addition(model: NeuralNet, train_loader, loss_fun, position, module_type, config):
    # Create a deep copy of the model
    model_copy = copy.deepcopy(model)
    model_copy.insert_layer(position, module_type, **config)

    n_p = model_copy.compute_natural_expansion_score(train_loader, loss_fun)

    # Compute the L1 penalty (sum of absolute values of parameters) times the regularization coefficient
    l1_penalty = L1_REG * sum(p.abs().sum() for p in model_copy.parameters())

    # Subtract the L1 penalty from the natural expansion score
    return n_p - l1_penalty

def calculate_insertion_benefits(model: NeuralNet, input_shape, possible_modules, module_config, train_loader, loss_fun):
    n_c = model.compute_natural_expansion_score(train_loader, loss_fun)  # Current expansion score
    benefits = []
    x = torch.rand((1, *input_shape)).to(device)  # Dummy input for shape inference
    x = model.input_layer(x)

    # Position before the output module
    position_before_output = len(model.layers)

    # Single Changes
    for i in range(0, position_before_output + 1, 2):  # Count by twos to skip over activation layers
        for module_type in possible_modules:
            local_config = module_config[module_type.__name__].copy()  # Make a local copy of the configuration

            # Adjust out_channels for Conv2d layers if adding before output module
            if module_type.__name__ == 'Conv2d' and i == position_before_output:
                local_config['out_channels'] = [FIXED_OUTPUT_CHANNELS]

            configs = list(itertools.product(*local_config.values()))
            for config in configs:
                config_dict = {k: v for k, v in zip(local_config.keys(), config)}

                # Set in_channels for Conv2d layers
                if module_type.__name__ == 'Conv2d':
                    config_dict['in_channels'] = x.shape[1]
                config_dict = update_padding_for_module(module_type, config_dict)
                n_p = calculate_expansion_score_with_addition(model, train_loader, loss_fun, i, module_type, config_dict)
                # print("Penalized n_p:", n_p.item())
                # print("n_c:", n_c)
                ratio = n_p / n_c
                print("n_p/n_c ", ratio.item())
                if ratio > TAU:
                    benefits.append((ratio, i, module_type, config_dict))

        # Forward pass through the next convolutional and activation layers
        if i < len(model.layers):
            x = model.layers[i](x)  # Convolutional layer
            if i+1 < len(model.layers):
                x = model.layers[i+1](x)  # Activation layer

    # Paired Changes
    return benefits

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fun = nn.BCEWithLogitsLoss()

    model.train()  # Set the model to training mode
    correct_train = 0
    total_train = 0

    for epoch in range(0, EPOCHS):
        for input, target in train_loader:  
            input, target = input.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fun(output, target)
            predicted = torch.sigmoid(output).round()  # Getting the binary predictions
            correct_train += (predicted == target).sum().item()
            total_train += target.size(0)
            loss.backward()
            optimizer.step()
        train_acc = 100 * correct_train / total_train

        # calculate validation scores
        model.eval()  # Set the model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss_total = 0
        for val_input, val_target in val_loader:  
            val_input, val_target = val_input.to(device).float(), val_target.to(device).float()
            optimizer.zero_grad()
            val_output = model(val_input)
            val_loss = loss_fun(val_output, val_target)
            val_loss_total += val_loss.item()
            predicted_val = torch.sigmoid(val_output).round()
            correct_val += (predicted_val == val_target).sum().item()
            total_val += val_target.size(0)

        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss_total / len(val_loader)
        num_params = model.count_parameters()
        wandb.log({"Train Loss": loss, "Train Accuracy": train_acc, "Val Loss": avg_val_loss, "Val Accuracy": val_acc, "Model Size (Params)": num_params})
        
        print(f'Epoch: {epoch} Train Loss: {loss.item()} Train Acc: {train_acc:.2f}% Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}% Number of Parameters: {num_params}')

        if epoch % 10 == 0:
            print("Model Summary:")
            summary(model, input_size=INPUT_SHAPE)


        # Calculate benefits for all possible insertions
        benefits = calculate_insertion_benefits(model, INPUT_SHAPE, POSSIBLE_SINGLE_MODULES, single_layer_configs, train_loader, loss_fun)

        # Select the best action if any are benefitial
        if benefits and len(benefits):
            _, position, best_module, best_config = select_best_action(benefits)
            # Insert the new layer
            model.insert_layer(position, best_module, **best_config)

        # Update the next layer after that one as necessary
        # TODO For this we need to adapt that next layer to accept the number of feature maps outputted by the newly added layer.
        # Can we do that without deleting the parameters we already learned and recreating it? 
        # Maybe we can copy the learned kernels and biases instead, and let the newly added ones to make shapes match be initalized randomly, then they get learned all together.

train()
