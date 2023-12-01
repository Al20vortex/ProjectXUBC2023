import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from torchsummary import summary

# Define our hyperparameters
INPUT_SHAPE = (3, 64, 64)
OUTPUT_SHAPE = (1,)
POSSIBLE_MODULES = [nn.Conv2d, nn.MaxPool2d]  # Added MaxPool2d for diversity

# Set up hyperparameters for each possible module
module_config = {
    "Conv2d": {
        "out_channels": [64, 128, 256],  # Adjust as per your requirement
        "kernel_size": [3, 5],
        "stride": [1, 2],
    },
    "MaxPool2d": {
        "kernel_size": [2, 3],
        "stride": [2],
    }
}

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
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layers = nn.ModuleList()

        # output layer placeholder
        self.out = None
        self.adapt_output_layer(64, INPUT_SHAPE[1], INPUT_SHAPE[2])

    def adapt_output_layer(self, num_feature_maps, x_dim, y_dim):
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_feature_maps * x_dim * y_dim, OUTPUT_SHAPE[0]),
            nn.Sigmoid()
        ).to(device)

    def insert_layer(self, position, module: nn.Module, **kwargs):
        if position < 0 or position > len(self.layers):
            raise ValueError("Invalid position to insert layer")
        
        new_layer = module(**kwargs).to(device)
        self.layers.insert(position, new_layer)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x.view(x.size(0))
      
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# TODO params needs to be useful information based off we decide to expand. For now we will simply expand every second epoch
def need_to_expand(params, epoch):
    """Checks if the model can benefit from expansion based on natural gradients"""
    if epoch % 2 == 0:
        return True
    else:
        return False

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

def calculate_insertion_benefits(model, input_shape, possible_modules, module_config):
    benefits = []
    x = torch.rand((1, *input_shape)).to(device)  # Dummy input for shape inference
    x = model.input_layer(x)
    
    for i in range(len(model.layers) + 1):
        for module_type in possible_modules:
            # Generate all combinations of configurations for the module
            configs = list(itertools.product(*module_config[module_type.__name__].values()))
            for config in configs:
                # Create a dictionary of arguments for the module constructor
                config_dict = {k: v for k, v in zip(module_config[module_type.__name__].keys(), config)}
                # print(f"config_dict: {config_dict}")
                if module_type.__name__ == 'Conv2d':
                    # Here we set the correct in_channels for convolution modules because theres always only one possibility based on the prev layer's output shape
                    config_dict['in_channels'] = x.shape[1]

                # Must also calculate the padding based on the stride and the kernel size.
                config_dict = update_padding_for_module(module_type, config_dict)
                
                # Insert hypothetical module
                hypothetical_module = module_type(**config_dict).to(device)

                # print(f"hypothetical_module: {hypothetical_module}")
                hypothetical_x = hypothetical_module(x)
                # print(f"hypothetical_x shape: {hypothetical_x.shape}")
                # Calculate benefit score (stub)
                benefit_score = get_module_benefit({}, hypothetical_module)
                
                # Store the benefit along with position and configuration
                benefits.append((benefit_score, i, module_type, config_dict))
        
        # Forward pass through the next actual layer
        if i < len(model.layers):
            x = model.layers[i](x)

    return benefits

def train():
    model = NeuralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.BCELoss()

    for epoch in range(1000):
        for input, target in train_loader:  
            input, target = input.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()

        # Print the loss and parameter count at the end of each epoch
        param_count = model.count_parameters()
        print(f'Epoch: {epoch} Loss: {loss.item()} Number of Parameters: {param_count}')

        if epoch % 10 == 0:
            print("Model Summary:")
            summary(model, input_size=INPUT_SHAPE)
        # Example usage in the train function:
        if need_to_expand({}, epoch):
            # Calculate benefits for all possible insertions
            benefits = calculate_insertion_benefits(model, INPUT_SHAPE, POSSIBLE_MODULES, module_config)

            # Select the best action
            _, position, best_module, best_config = select_best_action(benefits)

            # Insert the new layer
            model.insert_layer(position, best_module, **best_config)

            # Update the next layer after that one as necessary
            # TODO For this we need to adapt that next layer to accept the number of feature maps outputted by the newly added layer.
            # Can we do that without deleting the parameters we already learned and recreating it? 
            # Maybe we can copy the learned kernels and biases instead, and let the newly added ones to make shapes match be initalized randomly, then they get learned all together.

train()
